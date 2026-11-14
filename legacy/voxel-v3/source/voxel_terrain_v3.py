"""Read real Minecraft 1.20.1 block volumes, including underground geometry."""
from collections import Counter, OrderedDict
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import zlib

import numpy as np
import nbtlib

Y_MIN, Y_MAX = -64, 320
CLASS_NAMES = ['air','water','lava','stone','deepslate','soil','grass_moss','sand',
    'sandstone','gravel','snow','ice','terracotta','wood','leaves','plants',
    'calcite_dripstone','basalt_blackstone','ore','bedrock','amethyst','other_solid']
CLASS_SCHEMA = 'minecraft-semantic-voxel22-v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def semantic(name):
    s=name.split(':')[-1]
    if s in ('air','cave_air','void_air'):return 0
    if s in ('water','bubble_column'):return 1
    if s=='lava':return 2
    if s=='bedrock':return 19
    if s.endswith('_ore') or s=='ancient_debris':return 18
    if 'deepslate' in s:return 4
    if s in ('dirt','coarse_dirt','rooted_dirt','mud','packed_mud','clay','farmland','dirt_path'):return 5
    if s in ('grass_block','moss_block','mycelium','podzol'):return 6
    if s in ('sand','red_sand','suspicious_sand'):return 7
    if 'sandstone' in s:return 8
    if s in ('gravel','suspicious_gravel'):return 9
    if s in ('snow','snow_block','powder_snow'):return 10
    if s in ('ice','packed_ice','blue_ice','frosted_ice'):return 11
    if 'terracotta' in s:return 12
    if s.endswith(('_log','_wood','_stem','_hyphae')) or s in ('bamboo_block','stripped_bamboo_block'):return 13
    if s.endswith('_leaves') or s in ('nether_wart_block','warped_wart_block'):return 14
    if any(x in s for x in ('grass','fern','flower','tulip','sapling','roots','vine','lichen','seagrass','kelp','mushroom','azalea','dripleaf')) or s in ('dandelion','poppy','blue_orchid','allium','azure_bluet','lily_of_the_valley','wither_rose','sunflower','lilac','peony','dead_bush','sugar_cane','bamboo','cactus','lily_pad','moss_carpet','spore_blossom','hanging_roots','pink_petals'):return 15
    if s in ('calcite','dripstone_block','pointed_dripstone'):return 16
    if 'basalt' in s or 'blackstone' in s or s in ('magma_block','obsidian','crying_obsidian'):return 17
    if 'amethyst' in s:return 20
    if any(x in s for x in ('stone','granite','diorite','andesite','tuff','netherrack','end_stone')):return 3
    return 21


def packed_indices(palette, data, count, minimum_bits):
    if len(palette)==1:return np.zeros(count,dtype=np.uint16)
    if not palette:raise ValueError('Empty block or biome palette')
    bits=max(minimum_bits,(len(palette)-1).bit_length());per=64//bits
    words=np.asarray(data,dtype=np.int64).view(np.uint64)
    assert len(words)==(count+per-1)//per
    j=np.arange(count,dtype=np.int64)
    ids=((words[j//per] >> ((j%per)*bits).astype(np.uint64)) & np.uint64((1<<bits)-1)).astype(np.uint16)
    assert ids.max()<len(palette)
    return ids


class WorldReader:
    def __init__(self, world, expected_seed, cache_chunks=256, reference=None):
        self.world=Path(world);self.region=self.world/'region';self.seed=int(expected_seed)
        level=nbtlib.load(self.world/'level.dat',gzipped=True)
        assert int(level['Data']['WorldGenSettings']['seed'])==self.seed
        self.cache=OrderedDict();self.cache_chunks=cache_chunks;self.headers={}
        self.region_hashes={};self.unknown=Counter();self.biome_palette=['__missing__'];self.biome_ids={'__missing__':0}
        self.independent_checks=0;self.independent_biome_checks=0;self.checked_chunks=set()
        self.reference=reference;self.height_columns_verified=0

    def close(self):self.cache.clear()

    def chunk_nbt(self,x,z):
        path=self.region/f'r.{x//32}.{z//32}.mca'
        if path not in self.headers:
            if not path.is_file():raise KeyError((x,z))
            before=path.stat();self.headers[path]=path.read_bytes()[:4096]
            assert len(self.headers[path])==4096
            self.region_hashes[str(path)]={'sha256':sha(path),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns}
        index=(x%32)+(z%32)*32;entry=int.from_bytes(self.headers[path][4*index:4*index+4],'big');offset=entry>>8;sectors=entry&255
        if offset==0:raise KeyError((x,z))
        assert offset>=2 and sectors>0
        with path.open('rb') as f:
            f.seek(offset*4096);header=f.read(5);assert len(header)==5
            length,kind=struct.unpack('>IB',header);assert 1<=length<=sectors*4096-4
            if kind&128:raise NotImplementedError('External oversized chunks need an explicit reader')
            data=f.read(length-1);assert len(data)==length-1
        raw={1:gzip.decompress,2:zlib.decompress,3:lambda v:v}[kind](data)
        n=nbtlib.File.parse(io.BytesIO(raw));assert int(n['xPos'])==x and int(n['zPos'])==z
        assert str(n['Status']).split(':')[-1]=='full' and int(n['DataVersion'])==3465
        return n

    def chunk(self,x,z):
        key=(int(x),int(z))
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        n=self.chunk_nbt(*key);vox=np.zeros((384,16,16),dtype=np.uint8);bio=np.zeros((96,4,4),dtype=np.uint16)
        seen=set();legacy_ground=np.full((16,16),-32768,dtype=np.int16)
        from ground_extract import classify as legacy_classify
        for section in n['sections']:
            sy=int(section['Y']);start=sy*16-Y_MIN
            if not 0<=start<384:continue
            assert sy not in seen;seen.add(sy)
            states=section.get('block_states')
            if states:
                names=[str(v['Name']) for v in states['palette']];classes=np.array([semantic(v) for v in names],dtype=np.uint8)
                for name,k in zip(names,classes):
                    if k==21:self.unknown[name]+=1
                ids=packed_indices(names,states.get('data',[]),4096,4)
                vox[start:start+16]=classes[ids].reshape(16,16,16)
                if self.reference is not None:
                    terrain=np.array([legacy_classify(v)>1 for v in names])[ids].reshape(16,16,16)
                    present=terrain.any(0);top=sy*16+15-terrain[::-1].argmax(0)
                    legacy_ground[present]=np.maximum(legacy_ground[present],top[present])
                if key not in self.checked_chunks:
                    # Independent scalar indexing catches word packing and YZX ordering mistakes.
                    for j in (0,1,15,16,255,256,2048,4095):
                        if len(names)==1:index=0
                        else:
                            bits=max(4,(len(names)-1).bit_length());per=64//bits
                            word=int(states['data'][j//per])&((1<<64)-1)
                            index=(word>>((j%per)*bits))&((1<<bits)-1)
                        yy,rem=divmod(j,256);zz,xx=divmod(rem,16)
                        assert int(vox[start+yy,zz,xx])==semantic(names[index]);self.independent_checks+=1
            biomes=section.get('biomes')
            if biomes:
                names=[str(v) for v in biomes['palette']];global_ids=[]
                for name in names:
                    if name not in self.biome_ids:self.biome_ids[name]=len(self.biome_palette);self.biome_palette.append(name)
                    global_ids.append(self.biome_ids[name])
                ids=packed_indices(names,biomes.get('data',[]),64,1)
                bio[start//4:start//4+4]=np.asarray(global_ids,dtype=np.uint16)[ids].reshape(4,4,4)
                if key not in self.checked_chunks:
                    for j in (0,3,4,15,16,31,32,63):
                        if len(names)==1:index=0
                        else:
                            bits=max(1,(len(names)-1).bit_length());per=64//bits
                            word=int(biomes['data'][j//per])&((1<<64)-1)
                            index=(word>>((j%per)*bits))&((1<<bits)-1)
                        yy,rem=divmod(j,16);zz,xx=divmod(rem,4)
                        assert int(bio[start//4+yy,zz,xx])==global_ids[index];self.independent_biome_checks+=1
        if self.reference is not None:
            origin=self.reference['origin_chunk'];dx=x-int(origin[0]);dz=z-int(origin[1])
            assert dx>=0 and dz>=0 and self.reference['valid_chunks'][dz,dx]
            expected=self.reference['ground'][dz*16:dz*16+16,dx*16:dx*16+16]
            assert np.array_equal(legacy_ground,expected),'3D palette decode differs from independently archived surface heights'
            self.height_columns_verified+=256
        self.checked_chunks.add(key);self.cache[key]=(vox,bio)
        if len(self.cache)>self.cache_chunks:self.cache.popitem(last=False)
        return vox,bio

    def cube(self,origin):
        x0,y0,z0=map(int,origin);assert x0%16==z0%16==0 and y0%4==0 and Y_MIN<=y0<=Y_MAX-64
        v=np.empty((64,64,64),dtype=np.uint8);b=np.empty((16,16,16),dtype=np.uint16)
        for dz in range(4):
            for dx in range(4):
                chunk,bio=self.chunk(x0//16+dx,z0//16+dz);y=y0-Y_MIN
                v[:,dz*16:dz*16+16,dx*16:dx*16+16]=chunk[y:y+64]
                b[:,dz*4:dz*4+4,dx*4:dx*4+4]=bio[y//4:y//4+16]
        return v,b

    def assert_sources_unchanged(self):
        for name,entry in self.region_hashes.items():
            p=Path(name);s=p.stat();assert s.st_size==entry['bytes'] and s.st_mtime_ns==entry['mtime_ns'] and sha(p)==entry['sha256']


def cave_summary(voxels,ground,origin):
    x,y,z=map(int,origin)
    underground=(np.arange(y,y+64)[:,None,None] <= ground[None]-4)
    cave=(voxels==0)&underground
    solid=~np.isin(voxels,[0,1,2,15])
    return {'air_voxels':int((voxels==0).sum()),'water_voxels':int((voxels==1).sum()),
        'lava_voxels':int((voxels==2).sum()),'solid_voxels':int(solid.sum()),
        'underground_positions':int(underground.sum()),'underground_air_voxels':int(cave.sum()),
        'class_counts':np.bincount(voxels.ravel(),minlength=len(CLASS_NAMES)).tolist()}
