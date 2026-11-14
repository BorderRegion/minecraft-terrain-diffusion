"""Decode rich heads into catalogue block states while preserving semantic geometry."""
import numpy as np


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    result = np.asarray(value)
    assert np.isfinite(result).all(), 'Nonfinite decoder output'
    return result


def decode_catalogue(prediction, vocabulary, max_positions=32768):
    """Select semantic class, compatible block name, then a recorded property tuple.

    This is a constrained projection, not proof of physical stability or visual
    quality. Report every changed raw name and every repaired property tuple.
    Arrays use B,Y,Z,X; biome arrays retain their four-block cell resolution.
    """
    assert max_positions > 0
    semantic_logits = array(prediction['semantic'])
    name_logits = array(prediction['names'])
    property_logits = array(prediction['properties'])
    biome_logits = array(prediction['biomes'])
    assert semantic_logits.ndim == name_logits.ndim == property_logits.ndim == biome_logits.ndim == 5
    shape = semantic_logits.shape[:1] + semantic_logits.shape[2:]
    assert name_logits.shape[:1] + name_logits.shape[2:] == shape
    assert property_logits.shape[:1] + property_logits.shape[2:] == shape
    names = vocabulary['names']; palette = vocabulary['state_palette']
    assert semantic_logits.shape[1] == 22 and biome_logits.shape[0] == semantic_logits.shape[0]
    assert names[0] is None and palette[0] is None and vocabulary['biomes'][0] is None
    assert len(palette) < 65536 and len(names) == name_logits.shape[1]
    assert len(vocabulary['biomes']) == biome_logits.shape[1] < 65536
    sem = semantic_logits.argmax(1).astype(np.uint8).reshape(-1)
    nl = name_logits.transpose(1,0,2,3,4).reshape(len(names), -1)
    pl = property_logits.transpose(1,0,2,3,4).reshape(property_logits.shape[1], -1)
    raw_names = nl.argmax(0)
    name_semantics = np.asarray(vocabulary['name_semantics'])
    selected_names = np.empty_like(raw_names)
    for group in range(22):
        positions = np.flatnonzero(sem == group)
        candidates = np.flatnonzero(name_semantics == group)
        assert len(candidates) > 0, ('Missing semantic class in training names', group)
        for start in range(0, len(positions), max_positions):
            at = positions[start:start+max_positions]
            selected_names[at] = candidates[nl[np.ix_(candidates, at)].argmax(0)]
    name_ids = {name:i for i,name in enumerate(names) if name is not None}
    by_name = {}
    for index,state in enumerate(palette[1:],1):
        by_name.setdefault(name_ids[state['Name']], []).append(index)
    offsets = {}; ranges = {}; cursor = 0
    for key,values in sorted(vocabulary['properties'].items()):
        offsets[key] = {value:cursor+i for i,value in enumerate(values) if value is not None}
        ranges[key] = np.arange(cursor,cursor+len(values))
        cursor += len(values)
    assert cursor == pl.shape[0]
    states = np.empty(sem.size, np.uint16)
    repaired_properties = 0
    for name in np.unique(selected_names):
        candidates = np.asarray(by_name[int(name)])
        keys = sorted(palette[int(candidates[0])]['Properties'])
        assert all(sorted(palette[int(i)]['Properties']) == keys for i in candidates), 'Inconsistent property schema for block name'
        positions = np.flatnonzero(selected_names == name)
        for start in range(0, len(positions), max_positions):
            at = positions[start:start+max_positions]
            joint = np.zeros((len(candidates),len(at)),np.float32)
            raw_matches = np.ones((len(candidates),len(at)),bool)
            for key in keys:
                rows = np.asarray([offsets[key][palette[int(i)]['Properties'][key]] for i in candidates])
                joint += pl[np.ix_(rows,at)]
                available = ranges[key]
                raw = available[pl[np.ix_(available,at)].argmax(0)]
                raw_matches &= rows[:,None] == raw[None]
            states[at] = candidates[joint.argmax(0)]
            repaired_properties += int((~raw_matches.any(0)).sum())
    state_semantics = np.asarray([255]+[name_semantics[name_ids[s['Name']]] for s in palette[1:]],np.uint8)
    assert np.array_equal(state_semantics[states],sem)
    biomes = (biome_logits[:,1:].argmax(1)+1).astype(np.uint16)
    assert states.min() > 0 and states.max() < len(palette)
    report = dict(voxels=int(sem.size), raw_unknown_name_voxels=int((raw_names==0).sum()),
                  raw_name_semantic_disagreement_voxels=int((name_semantics[raw_names]!=sem).sum()),
                  projected_name_changes=int((raw_names!=selected_names).sum()),
                  repaired_non_catalogue_property_tuples=repaired_properties,
                  raw_unknown_biome_cells=int((biome_logits.argmax(1)==0).sum()),
                  semantic_geometry_bitwise_preserved=True,
                  scope='Constrained training-catalogue decode. Native registry, neighbour, fluid and full-world quality checks remain required.')
    return dict(semantic=sem.reshape(shape),names=selected_names.reshape(shape),
                block_states=states.reshape(shape),biomes=biomes,projection=report)
