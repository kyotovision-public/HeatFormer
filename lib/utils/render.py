import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
import pyrender
import numpy as np

def render_mesh(img, mesh, face, K, c=0.9, side=False):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    if side:
        rot = trimesh.transformations.rotation_matrix(
            np.radians(90), [0, 1, 0])
        mesh.apply_transform(rot)
    rot = trimesh.transformations.rotation_matrix(
    np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    # material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(0.85882353,  0.74117647,  0.65098039, 1.0))
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    if type(K)==dict and 'focal' in K and 'princpt' in K:
        cam_param = K
    else:
        K = K.reshape(3,3)
        cam_param = {}
        cam_param['focal'] = np.array([K[0,0], K[1,1]])
        cam_param['princpt'] = np.array([K[0,2], K[1,2]])
    
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    if side:
        return rgb
    else:
        # save to image
        img = rgb * valid_mask + img * (1-valid_mask)
        return img
    
def render_mesh_mod(img, mesh, face, K, trans, c=0.9, side=False, rot_angle=90, debug=False):
    trans[0] *= -1

    if debug:
        import pdb;pdb.set_trace()

    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    if side:
        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), [0, 1, 0])
        mesh.apply_transform(rot)
    rot = trimesh.transformations.rotation_matrix(
    np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, c, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    if type(K)==dict and 'focal' in K and 'princpt' in K:
        cam_param = K
    else:
        K = K.reshape(3,3)
        cam_param = {}
        cam_param['focal'] = np.array([K[0,0], K[1,1]])
        cam_param['princpt'] = np.array([K[0,2], K[1,2]])
    
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = trans
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera, pose=camera_pose)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    if side:
        return rgb
    else:
        # save to image
        img = rgb * valid_mask + img * (1-valid_mask)
        return img
    
def look_at_matrix(camera_position, target, up=[0, 1, 0]):
    """
    Creates a transformation matrix for a camera to look at a target point.
    
    Parameters:
    - camera_position: The position of the camera (3-element list or np.array).
    - target: The point the camera looks at (3-element list or np.array).
    - up: The "up" vector for the camera (3-element list or np.array).

    Returns:
    - A 4x4 transformation matrix.
    """
    camera_position = np.array(camera_position, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    forward = target - camera_position
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    true_up = np.cross(forward, right)
    
    rotation = np.eye(4)
    rotation[:3, :3] = np.stack([right, true_up, forward], axis=-1)
    
    translation = np.eye(4)
    translation[:3, 3] = -camera_position

    return rotation @ translation

def render_around_view(mesh):
    # Step 1: Create a 3D mesh object
    mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=3)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.7, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    # Step 2: Create a scene and add the mesh
    scene = pyrender.Scene()
    scene.add(mesh)

    # Step 3: Add a light source
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=np.eye(4))

    # Step 4: Define camera position, target, and look-at pose
    camera_position = [0.3, 0.3, 0.3]  # Camera position in 3D space
    target = [0.0, 0.0, 0.0]          # The object center
    camera_pose = look_at_matrix(camera_position, target)

    # Step 5: Create a camera and add it to the scene
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=camera_pose)

    # Step 6: Render the scene
    renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)
    rgb, depth = renderer.render(scene)

    return rgb[:,:,:3].astype(np.float32)
