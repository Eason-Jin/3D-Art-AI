import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt

def render_model(model_path):
    mesh = trimesh.load(model_path)

    scene = pyrender.Scene()
    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_node)

    angles = [i for i in range(0, 360, 30)]
    renders = []

    for angle in angles:
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        
        zoom = 5.0
        theta = np.radians(angle)
        camera_pose = np.array([
            [np.cos(theta), 0, np.sin(theta), zoom * np.sin(theta)],
            [0, 1, 0, 0.5],
            [-np.sin(theta), 0, np.cos(theta), zoom * np.cos(theta)],
            [0, 0, 0, 1]
        ])
        
        camera_node = scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        light_node = scene.add(light, pose=camera_pose)
        
        renderer = pyrender.OffscreenRenderer(512, 512)
        color, _ = renderer.render(scene)
        # plt.imsave(f"obj/renders/render_{angle}.png", color)
        renders.append(color)

        scene.remove_node(camera_node)
        scene.remove_node(light_node)

    renderer.delete()
    return renders

print(len(render_model("obj/monkey.obj")))