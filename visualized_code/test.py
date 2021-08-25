# need pay attention to the method into the keyshot！
# we can render the image sample by sample!
# we need delete the data?
#1. set the input mode?
#2. set the render mode?
#3. and based on the render environment to render the images! batch render the data.
import os, re
src_path_files = [r'F:\Ubuntu_jack\windows_shapr_papers\whole_data\rebuttal\srcnew.obj']

# set some propeties  of the input files!
# we need set suitable driver devices!
# lux.setRenderEngine(lux.RENDER_ENGINE_PRODUCT_GPU)
opts = lux.getRenderOptions()
opts.setThreads(8)
opts.setRayBounces(64)
opts = lux.getImportOptions()

# For render properities, we need pay attention!
opts['accurate_tessellation'] = True
opts["snap_to_ground"] = False
opts["adjust_environment"] = True
opts["adjust_camera_look_at"] = False
opts["applyLibraryMaterials"] = False
opts["camera_import"] = True
opts['center_geometry'] = False
opts['compute_normals'] = False
opts['frame'] = 0
opts['geometry_scale'] = 10
opts['geometry_units'] = 1000.0
lux.importFile(tgt_path, opts = opts)
root = lux.getSceneTree()

#Input the file and set the material!
for node in root.find(""):
    tp_node_name = node.getName()
    if '.obj' in tp_node_name and 'Source' not in tp_node_name and 'Target' not in tp_node_name and 'Source1' not in tp_node_name:
        node.setMaterial('Target')
        node.setName('Target_obj_cases')

for src_path in src_path_files:
    if 'tar' not in src_path:
        lux.importFile(src_path, opts=opts)

        root = lux.getSceneTree()

    for node in root.find(""):
        tp_src_name = node.getName()
        if '.obj' in tp_src_name and 'Source' not in tp_src_name and 'Target' not in tp_src_name and 'Source1' not in tp_src_name:
            print(tp_src_name)
            node.setMaterial('Source')
            # node.setName('Target_obj_cases')
        # 
    # for each obj status, we need prepare the data!
    # Path of the Render results!
    lux.renderImage(src_path.replace('.obj', '2.png', 1), width = 2000, height = 1500)
    for node in root.find(""):
        tp_src_name = node.getName()
        if '.obj' in tp_src_name and 'Source' not in tp_src_name and 'Target' not in tp_src_name and 'Source1' not in tp_src_name:
            node.hide()
