package net.gavrix32.app.scenes;

import net.gavrix32.engine.graphics.*;
import net.gavrix32.engine.linearmath.Vector3f;

public class BVHTest {
    private final Scene scene;
    private static Model model;
    public static BoundingVolumeHierarchy bvh;

    public BVHTest() {
        scene = new Scene();
        scene.setName("BVH Test");
        //scene.setCamera(new Vector3f(-25, 90, -150));
        scene.setCamera(new Vector3f(-25, 200, 500)).rotateY(180); // breakfast_room
        //scene.setCamera(new Vector3f(0, 100, 300)).rotateY(180); // cornell_box
        //scene.setCamera(new Vector3f(-80, 0, 0)).rotateY(90); // dragon
        scene.setSky("textures/sky/quarry_cloudy_2k.hdr").setMaterial(false, 1, 1, 1, false);
        //scene.getSky().setColor(1, 1, 1).setMaterial(true, 1, 1, 1, false);
        model = new Model("models/xyzrgb_dragon.obj", 100.0f);
        bvh = new BoundingVolumeHierarchy();
        bvh.build(model.getTriangles());
    }

    public static Model getModel() {
        return model;
    }

    public Scene getScene() {
        return scene;
    }
}