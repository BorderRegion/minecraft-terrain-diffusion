"""Bounded rich-world playback with explicit high-priority dimension generation settings."""
from collections import defaultdict
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import time
import zlib
import numpy as np

ROOT=Path('/root/autodl-tmp/minecraft-terrain')
if str(ROOT/'vendor') not in sys.path:sys.path.append(str(ROOT/'vendor'))
import nbtlib
from nbtlib import Compound,Int,Long,Byte,String,List,LongArray,Short
from voxel_training_data_v3 import sha


def packed_data(indices,palette_size,minimum_bits):
    flat=np.asarray(indices).reshape(-1)
    assert palette_size>0 and flat.min()>=0 and flat.max()<palette_size
    if palette_size==1:return None
    bits=max(minimum_bits,(palette_size-1).bit_length());per=64//bits
    words=np.zeros((len(flat)+per-1)//per,np.uint64)
    for slot in range(per):
        values=flat[slot::per].astype(np.uint64)
        words[:len(values)]|=values<<np.uint64(slot*bits)
    return LongArray(words.view(np.int64))


def state_container(values,palette):
    unique,inverse=np.unique(values,return_inverse=True)
    entries=[]
    for index in unique:
        state=palette[int(index)]
        entry=Compound({'Name':String(state['Name'])})
        if state['Properties']:entry['Properties']=Compound({key:String(value) for key,value in state['Properties'].items()})
        entries.append(entry)
    result=Compound({'palette':List[Compound](entries)})
    data=packed_data(inverse,len(unique),4)
    if data is not None:result['data']=data
    return result


def biome_container(values,palette):
    unique,inverse=np.unique(values,return_inverse=True)
    result=Compound({'palette':List[String]([String(palette[int(index)]) for index in unique])})
    data=packed_data(inverse,len(unique),1)
    if data is not None:result['data']=data
    return result


def chunk_file(cx,cz,states,biomes,state_palette,biome_palette,template):
    assert states.shape==(384,16,16) and biomes.shape==(96,4,4)
    sections=[]
    for section in range(24):
        sections.append(Compound({'Y':Byte(section-4),
            'block_states':state_container(states[section*16:section*16+16],state_palette),
            'biomes':biome_container(biomes[section*4:section*4+4],biome_palette)}))
    fields={'DataVersion':Int(3465),'xPos':Int(cx),'zPos':Int(cz),'yPos':Int(-4),
            'Status':String('minecraft:full'),'LastUpdate':Long(0),'InhabitedTime':Long(0),
            'sections':List[Compound](sections),'block_entities':List[Compound]([]),
            'block_ticks':List[Compound]([]),'fluid_ticks':List[Compound]([]),
            'PostProcessing':List[List[Short]]([List[Short]([]) for _ in range(24)]),
            'structures':Compound({'starts':Compound({}),'References':Compound({})}),
            'isLightOn':Byte(0)}
    # Keep the terrain profile's version marker so a mod does not mistake the
    # explicitly exported full chunks for old terrain requiring retro-blending.
    if 'tectonic:blending_version' in template:fields['tectonic:blending_version']=copy.deepcopy(template['tectonic:blending_version'])
    # Omit height maps and saved light. The native server must recompute them;
    # the post-load audit checks the actual saved result instead of inventing maps.
    result=nbtlib.File(fields);stream=io.BytesIO();result.write(stream)
    compressed=zlib.compress(stream.getvalue(),6)
    record=struct.pack('>IB',len(compressed)+1,2)+compressed
    sectors=(len(record)+4095)//4096
    assert sectors<=255,'Oversized external chunks require an explicit writer'
    return record+b'\0'*(sectors*4096-len(record)),sectors


def export_world(destination,states,biomes,vocabulary,origin_xyz,level_template,chunk_template,seed,scope,outside_generator="void"):
    destination=Path(destination);level_template=Path(level_template)
    assert not destination.exists() and not destination.is_symlink()
    states=np.asarray(states);biomes=np.asarray(biomes);x0,y0,z0=map(int,origin_xyz)
    assert states.ndim==3 and states.dtype==np.uint16
    height,length,width=states.shape
    assert all(v>0 and v%16==0 for v in [height,length,width])
    assert x0%16==z0%16==y0%16==0 and -64<=y0 and y0+height<=320
    assert biomes.shape==(height//4,length//4,width//4) and biomes.dtype==np.uint16
    state_palette=copy.deepcopy(vocabulary['state_palette']);biome_palette=list(vocabulary['biomes'])
    assert state_palette[0] is None and biome_palette[0] is None
    assert states.min()>0 and states.max()<len(state_palette)
    assert biomes.min()>0 and biomes.max()<len(biome_palette)
    air={'Name':'minecraft:air','Properties':{}}
    if air not in state_palette:state_palette.append(air)
    air_id=state_palette.index(air)
    if 'minecraft:plains' not in biome_palette:biome_palette.append('minecraft:plains')
    outside_biome=biome_palette.index('minecraft:plains')
    temporary=destination.with_name(destination.name+'.incomplete');temporary.mkdir(parents=True,exist_ok=False)
    level=nbtlib.load(level_template,gzipped=True);assert int(level['Data']['DataVersion'])==3465
    data=level['Data'];data['WorldGenSettings']['seed']=Long(seed)
    data['LevelName']=String('Rich voxel3D export');data['allowCommands']=Byte(1);data['GameType']=Int(1)
    data['SpawnX']=Int(x0+width//2);data['SpawnZ']=Int(z0+length//2);data['SpawnY']=Int(min(319,y0+height+8))
    data['LastPlayed']=Long(int(time.time()*1000));data['Difficulty']=Byte(0)
    data['raining']=Byte(0);data['thundering']=Byte(0);data['clearWeatherTime']=Int(1000000);data['DayTime']=Long(6000)
    for key,value in {'doMobSpawning':'false','randomTickSpeed':'0','doDaylightCycle':'false','doWeatherCycle':'false'}.items():
        data['GameRules'][key]=String(value)
    data.pop('Player',None);data['ScheduledEvents']=List[Compound]([])
    original_packs=list(map(str,data['DataPacks']['Enabled']))
    data['DataPacks']['Enabled']=List[String]([String('vanilla'),String('fabric'),String('file/voxel_model_playback')])
    data['DataPacks']['Disabled']=List[String]([])
    assert outside_generator in ('void','retain_template')
    if outside_generator=='void':
        data['WorldGenSettings']['dimensions']['minecraft:overworld']['generator']=Compound({
            'type':String('minecraft:flat'),
            'settings':Compound({'biome':String('minecraft:plains'),'features':Byte(0),'lakes':Byte(0),
                'structure_overrides':List[String]([]),
                'layers':List[Compound]([Compound({'height':Int(1),'block':String('minecraft:air')})])})})
    level.save(temporary/'level.dat',gzipped=True)
    if (level_template.parent/'datapacks').is_dir():shutil.copytree(level_template.parent/'datapacks',temporary/'datapacks')
    playback=temporary/'datapacks/voxel_model_playback'
    dimensions=playback/'data/minecraft/dimension';dimensions.mkdir(parents=True,exist_ok=False)
    (playback/'pack.mcmeta').write_text(json.dumps({'pack':{'pack_format':15,'description':'Bounded learned voxel region playback'}})+'\n')
    definition={'type':'minecraft:overworld','generator':{'type':'minecraft:flat',
        'settings':{'biome':'minecraft:plains','features':False,'lakes':False,
                    'structure_overrides':[],'layers':[{'height':1,'block':'minecraft:air'}]}}}
    (dimensions/'overworld.json').write_text(json.dumps(definition,indent=2)+'\n')
    region=temporary/'region';region.mkdir()
    groups=defaultdict(list)
    for dz in range(length//16):
        for dx in range(width//16):
            cx=x0//16+dx;cz=z0//16+dz;groups[(cx//32,cz//32)].append((cx,cz,dx,dz))
    chunks=0
    for (rx,rz),entries in sorted(groups.items()):
        path=region/('r.'+str(rx)+'.'+str(rz)+'.mca')
        locations=bytearray(4096);timestamps=bytearray(4096);sector=2
        with path.open('xb') as handle:
            handle.write(b'\0'*8192)
            for cx,cz,dx,dz in sorted(entries,key=lambda value:((value[1]%32)*32+value[0]%32)):
                full=np.full((384,16,16),air_id,np.uint16)
                full_bio=np.full((96,4,4),outside_biome,np.uint16)
                y=y0+64
                full[y:y+height]=states[:,dz*16:dz*16+16,dx*16:dx*16+16]
                full_bio[y//4:(y+height)//4]=biomes[:,dz*4:dz*4+4,dx*4:dx*4+4]
                record,sectors=chunk_file(cx,cz,full,full_bio,state_palette,biome_palette,chunk_template)
                index=(cx%32)+(cz%32)*32
                locations[index*4:index*4+4]=((sector<<8)|sectors).to_bytes(4,'big')
                timestamps[index*4:index*4+4]=int(time.time()).to_bytes(4,'big')
                handle.write(record);sector+=sectors;chunks+=1
            handle.seek(0);handle.write(locations);handle.write(timestamps)
    metadata=dict(format='rich-voxel-anvil-export-v4',minecraft='1.20.1',data_version=3465,
                  seed=int(seed),origin_xyz=[x0,y0,z0],physical_shape_yzx=list(states.shape),chunks=chunks,
                  state_palette=state_palette,biome_palette=biome_palette,
                  template_level_sha256=sha(level_template),scope=scope,
                  original_generation_pack_ids=original_packs,playback_pack_ids=['vanilla','fabric','file/voxel_model_playback'],
                  playback_runtime='Minecraft1.20.1 plus FabricAPI and Terralith2.5.4; omit Tectonic/Lithostitched generation overrides in this isolated playback world.',
                  input_state_ids_sha256=hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest(),
                  input_biome_ids_sha256=hashlib.sha256(np.ascontiguousarray(biomes).tobytes()).hexdigest(),
                  source_code_sha256=sha(Path(__file__)),model_goal_complete=False,
                  outside_generator=outside_generator,
                  outside_input_volume='Air/plains within partial exported chunks. Unexported overworld chunks use the selected exterior generator; void disables natural feature writes into the bounded model region.',
                  limits=['No entities,block entities,scheduled block/fluid ticks or POI are represented by the trained input.',
                          'Missing height/light data requires actual native recomputation and post-load validation.',
                          'Correct file serialization is not a full-world aesthetics,fluid or cave-connectivity acceptance.'])
    metadata['files']={str(p.relative_to(temporary)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in temporary.rglob('*') if p.is_file()}
    with (temporary/'voxel-export.json').open('x') as handle:json.dump(metadata,handle,indent=2)
    temporary.rename(destination)
    return metadata
