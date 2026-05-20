package net.gavrix32.engine.graphics;

import net.gavrix32.app.scenes.BVHTest;
import net.gavrix32.engine.io.Window;
import net.gavrix32.engine.linearmath.Matrix4f;
import net.gavrix32.engine.linearmath.Vector3f;
import org.tinylog.Logger;

import java.nio.ByteBuffer;

import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL46C.*;
import static org.lwjgl.stb.STBImage.*;

public class Renderer {
    private static Scene scene;
    private static Shader pathtraceShader;
    private static Shader atrousShader;
    private static Shader presentShader;
    private static int accumulatedSamples = 0;
    private static int frameIndex = 0;
    private static Texture albedoTexture, positionTexture;
    private static final Texture[] normalTexture = new Texture[2];
    private static final Texture[] pathtraceTexture = new Texture[2];
    private static final Texture[] atrousTexture = new Texture[2];
    private static int prevPt = 1, currPt = 0;
    private static int prevAtr = 1, currAtr = 0;
    public static Vector3f triangles_offset = new Vector3f(0, 0, 0);
    public static Vector3f triangles_rotation = new Vector3f(0, 0, 0);

    // Config variables
    private static int samples, maxAccumulatedSamples, bounces;
    private static float gamma, exposure, focusDistance, aperture, fov;
    private static boolean accumulation, temporalReprojection, temporalAntialiasing, atrousFilter, russianRoulette, checkerboardRendering;

    private static Texture modelTexture;

    public static void init() {
        samples = Config.getInt("samples");
        maxAccumulatedSamples = Config.getInt("max_accumulated_samples");
        bounces = Config.getInt("bounces");
        gamma = Config.getFloat("gamma");
        exposure = Config.getFloat("exposure");
        focusDistance = Config.getFloat("focus_distance");
        aperture = Config.getFloat("aperture");
        fov = Config.getFloat("fov");
        accumulation = Config.getBoolean("accumulation");
        temporalReprojection = Config.getBoolean("temporal_reprojection");
        temporalAntialiasing = Config.getBoolean("temporal_antialiasing");
        atrousFilter = Config.getBoolean("atrous_filter");
        russianRoulette = Config.getBoolean("russian_roulette");

        Quad.init();

        pathtraceShader = new Shader();
        pathtraceShader.load("shaders/pathtrace.cs.glsl", GL_COMPUTE_SHADER);
        pathtraceShader.initProgram();

        atrousShader = new Shader();
        atrousShader.load("shaders/atrous.cs.glsl", GL_COMPUTE_SHADER);
        atrousShader.initProgram();

        presentShader = new Shader();
        presentShader.load("shaders/quad.vs.glsl", GL_VERTEX_SHADER);
        presentShader.load("shaders/present.fs.glsl", GL_FRAGMENT_SHADER);
        presentShader.initProgram();

        scene = new Scene();

        positionTexture = new Texture();
        positionTexture.bind();
        positionTexture.texImage(GL_RGBA32F, Window.getWidth(), Window.getHeight(), GL_FLOAT, 0);
        positionTexture.linearFiltering();
        positionTexture.clampToEdge();
        glBindImageTexture(0, positionTexture.getId(), 0, false, 0, GL_READ_WRITE, GL_RGBA32F);
        positionTexture.unbind();

        albedoTexture = new Texture();
        albedoTexture.bind();
        albedoTexture.texImage(GL_RGBA32F, Window.getWidth(), Window.getHeight(), GL_FLOAT, 0);
        albedoTexture.linearFiltering();
        albedoTexture.clampToEdge();
        glBindImageTexture(1, albedoTexture.getId(), 0, false, 0, GL_READ_WRITE, GL_RGBA32F);
        albedoTexture.unbind();

        for (int i = 0; i < 2; i++) {
            normalTexture[i] = new Texture();
            normalTexture[i].bind();
            normalTexture[i].texImage(GL_RGBA32F, Window.getWidth(), Window.getHeight(), GL_FLOAT, 0);
            normalTexture[i].linearFiltering();
            normalTexture[i].clampToEdge();
            glBindImageTexture(2, normalTexture[i].getId(), 0, false, 0, GL_READ_WRITE, GL_RGBA32F);
            normalTexture[i].unbind();

            pathtraceTexture[i] = new Texture();
            pathtraceTexture[i].bind();
            pathtraceTexture[i].texImage(GL_RGBA32F, Window.getWidth(), Window.getHeight(), GL_FLOAT, 0);
            pathtraceTexture[i].linearFiltering();
            pathtraceTexture[i].clampToEdge();
            glBindImageTexture(3, pathtraceTexture[i].getId(), 0, false, 0, GL_READ_WRITE, GL_RGBA32F);
            pathtraceTexture[i].unbind();

            atrousTexture[i] = new Texture();
            atrousTexture[i].bind();
            atrousTexture[i].texImage(GL_RGBA32F, Window.getWidth(), Window.getHeight(), GL_FLOAT, 0);
            atrousTexture[i].linearFiltering();
            atrousTexture[i].clampToEdge();
            glBindImageTexture(4, atrousTexture[i].getId(), 0, false, 0, GL_READ_WRITE, GL_RGBA32F);
            atrousTexture[i].unbind();
        }

        // Send BVH data to GPU
        BoundingVolumeHierarchy bvh = BVHTest.bvh;

        float[] bvh_node_data = new float[12 * bvh.nodes.size()];
        int index = 0;
        for (int i = 0; i < bvh.nodes.size(); i++) {
            bvh_node_data[index++] = bvh.nodes.get(i).bounds.min.x;
            bvh_node_data[index++] = bvh.nodes.get(i).bounds.min.y;
            bvh_node_data[index++] = bvh.nodes.get(i).bounds.min.z;
            bvh_node_data[index++] = 0.0f;
            bvh_node_data[index++] = bvh.nodes.get(i).bounds.max.x;
            bvh_node_data[index++] = bvh.nodes.get(i).bounds.max.y;
            bvh_node_data[index++] = bvh.nodes.get(i).bounds.max.z;
            bvh_node_data[index++] = 0.0f;
            bvh_node_data[index++] = bvh.nodes.get(i).triangleStartIndex;
            bvh_node_data[index++] = bvh.nodes.get(i).trianglesCount;
            bvh_node_data[index++] = bvh.nodes.get(i).childIndex;
            bvh_node_data[index++] = 0.0f;
        }
        int bvh_nodes_ssbo = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bvh_nodes_ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bvh_node_data, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bvh_nodes_ssbo);

        float[] bvh_triangles_data = new float[24 * bvh.triangles.size()];
        index = 0;
        for (int i = 0; i < bvh.triangles.size(); i++) {
            bvh_triangles_data[index++] = bvh.triangles.get(i).v1.x;
            bvh_triangles_data[index++] = bvh.triangles.get(i).v1.y;
            bvh_triangles_data[index++] = bvh.triangles.get(i).v1.z;
            bvh_triangles_data[index++] = 0.0f;
            bvh_triangles_data[index++] = bvh.triangles.get(i).v2.x;
            bvh_triangles_data[index++] = bvh.triangles.get(i).v2.y;
            bvh_triangles_data[index++] = bvh.triangles.get(i).v2.z;
            bvh_triangles_data[index++] = 0.0f;
            bvh_triangles_data[index++] = bvh.triangles.get(i).v3.x;
            bvh_triangles_data[index++] = bvh.triangles.get(i).v3.y;
            bvh_triangles_data[index++] = bvh.triangles.get(i).v3.z;
            bvh_triangles_data[index++] = 0.0f;
            bvh_triangles_data[index++] = bvh.triangles.get(i).uv1.x;
            bvh_triangles_data[index++] = bvh.triangles.get(i).uv1.y;
            bvh_triangles_data[index++] = 0.0f;
            bvh_triangles_data[index++] = 0.0f;
            bvh_triangles_data[index++] = bvh.triangles.get(i).uv2.x;
            bvh_triangles_data[index++] = bvh.triangles.get(i).uv2.y;
            bvh_triangles_data[index++] = 0.0f;
            bvh_triangles_data[index++] = 0.0f;
            bvh_triangles_data[index++] = bvh.triangles.get(i).uv3.x;
            bvh_triangles_data[index++] = bvh.triangles.get(i).uv3.y;
            bvh_triangles_data[index++] = 0.0f;
            bvh_triangles_data[index++] = 0.0f;
        }
        int bvh_triangles_ssbo = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bvh_triangles_ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bvh_triangles_data, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bvh_triangles_ssbo);

        // Model texture
        //int[] width = new int[1], height = new int[1];
        /*byte[] bytes = Utils.loadBytes("models/breakfast_room/picture3.jpg");
        ByteBuffer buffer = BufferUtils.createByteBuffer(bytes.length);
        buffer.put(bytes).flip();*/
        /*stbi_set_flip_vertically_on_load(true);
        ByteBuffer data = stbi_load_from_memory(BVHTest.getModel().data, width, height, new int[1], 3);
        if (data == null) Logger.error("Failed to load texture: \"\"" + " " + stbi_failure_reason());*/

        /*modelTexture = new Texture();
        modelTexture.bind();
        modelTexture.linearFiltering();
        modelTexture.texImage(GL_RGB, width[0], height[0], GL_UNSIGNED_BYTE, data);
        modelTexture.unbind();*/
    }

    public static void render() {
        glClear(GL_COLOR_BUFFER_BIT);
        pathtraceShader.use();
        pathtraceShader.setMat4("prev_camera_rotation", scene.camera.getRotationMatrix());
        pathtraceShader.setVec3("prev_camera_position", scene.camera.getPosition());
        scene.camera.update();
        pathtraceShader.setMat4("camera_rotation", scene.camera.getRotationMatrix());
        pathtraceShader.setVec3("camera_position", scene.camera.getPosition());
        pathtraceShader.setFloat("time", (float) glfwGetTime());
        pathtraceShader.setInt("samples", samples);
        pathtraceShader.setInt("accumulated_samples", accumulatedSamples);
        pathtraceShader.setInt("frame_index", frameIndex);
        pathtraceShader.setInt("max_accumulated_samples", maxAccumulatedSamples);
        pathtraceShader.setInt("bounces", bounces);
        pathtraceShader.setBool("russian_roulette", russianRoulette);
        pathtraceShader.setFloat("fov", fov);
        pathtraceShader.setBool("temporal_reprojection", temporalReprojection);
        pathtraceShader.setBool("temporal_antialiasing", temporalAntialiasing);
        pathtraceShader.setBool("checkerboard_rendering", checkerboardRendering);
        pathtraceShader.setFloat("focus_distance", focusDistance);
        pathtraceShader.setFloat("aperture", aperture);
        pathtraceShader.setBool("sky_has_texture", scene.sky.hasTexture());
        pathtraceShader.setBool("enable_bvh", scene.getName().equals("BVH Test"));
        pathtraceShader.setBool("debug_bvh", Gui.debugBVH.get());
        pathtraceShader.setInt("bounds_test_threshold", Gui.boundsTestThreshold[0]);
        pathtraceShader.setInt("triangle_test_threshold", Gui.triangleTestThreshold[0]);
        if (scene.sky.hasTexture()) {
            glActiveTexture(GL_TEXTURE1);
            scene.sky.bindTexture();
            pathtraceShader.setInt("sky_texture", 1);
        } else {
            pathtraceShader.setVec3("sky.material.albedo", scene.sky.getColor());
        }
        pathtraceShader.setBool("sky.material.is_metal", scene.sky.getMaterial().isMetal());
        pathtraceShader.setFloat("sky.material.emission", scene.sky.getMaterial().getEmission());
        pathtraceShader.setFloat("sky.material.roughness", scene.sky.getMaterial().getRoughness());
        pathtraceShader.setBool("sky.material.is_glass", scene.sky.getMaterial().isGlass());
        pathtraceShader.setFloat("sky.material.IOR", scene.sky.getMaterial().getIOR());
        // Plane
        if (scene.plane != null) {
            pathtraceShader.setInt("plane.exists", 1);
            pathtraceShader.setBool("plane.checkerboard", scene.plane.isCheckerBoard());
            if (scene.plane.isCheckerBoard()) {
                pathtraceShader.setVec3("plane.color1", scene.plane.getFirstColor());
                pathtraceShader.setVec3("plane.color2", scene.plane.getSecondColor());
                pathtraceShader.setFloat("plane.scale", scene.plane.getScale());
            } else {
                pathtraceShader.setVec3("plane.material.albedo", scene.plane.getColor());
            }
            pathtraceShader.setFloat("plane.material.emission", scene.plane.getMaterial().getEmission());
            pathtraceShader.setFloat("plane.material.roughness", scene.plane.getMaterial().getRoughness());
            pathtraceShader.setBool("plane.material.is_glass", scene.plane.getMaterial().isGlass());
            pathtraceShader.setFloat("plane.material.IOR", scene.plane.getMaterial().getIOR());
            pathtraceShader.setBool("plane.material.is_metal", scene.plane.getMaterial().isMetal());
        } else {
            pathtraceShader.setInt("plane.exists", 0);
        }
        // Spheres
        BoundingBox sphBounds = new BoundingBox();
        pathtraceShader.setInt("spheres_count", scene.spheres.size());
        for (int i = 0; i < scene.spheres.size(); i++) {
            pathtraceShader.setVec3("spheres[" + i + "].position", scene.getSphere(i).getPos());
            pathtraceShader.setFloat("spheres[" + i + "].radius", scene.getSphere(i).getRadius());
            pathtraceShader.setVec3("spheres[" + i + "].material.albedo", scene.getSphere(i).getColor());
            pathtraceShader.setBool("spheres[" + i + "].material.is_metal", scene.getSphere(i).getMaterial().isMetal());
            pathtraceShader.setFloat("spheres[" + i + "].material.emission", scene.getSphere(i).getMaterial().getEmission());
            pathtraceShader.setFloat("spheres[" + i + "].material.roughness", scene.getSphere(i).getMaterial().getRoughness());
            pathtraceShader.setBool("spheres[" + i + "].material.is_glass", scene.getSphere(i).getMaterial().isGlass());
            pathtraceShader.setFloat("spheres[" + i + "].material.IOR", scene.getSphere(i).getMaterial().getIOR());
            sphBounds.addSphere(scene.getSphere(i));
        }
        pathtraceShader.setVec3("sph_bounds.min", sphBounds.min);
        pathtraceShader.setVec3("sph_bounds.max", sphBounds.max);
        // Boxes
        BoundingBox boxBounds = new BoundingBox();
        pathtraceShader.setInt("boxes_count", scene.boxes.size());
        for (int i = 0; i < scene.boxes.size(); i++) {
            pathtraceShader.setVec3("boxes[" + i + "].position", scene.boxes.get(i).getPos());
            scene.boxes.get(i).getRotationMatrix().rotate(
                    scene.boxes.get(i).getRot().x,
                    scene.boxes.get(i).getRot().y,
                    scene.boxes.get(i).getRot().z
            );
            pathtraceShader.setMat4("boxes[" + i + "].rotation", scene.getBox(i).getRotationMatrix());
            pathtraceShader.setVec3("boxes[" + i + "].scale", scene.getBox(i).getScale());
            pathtraceShader.setVec3("boxes[" + i + "].material.albedo", scene.getBox(i).getColor());
            pathtraceShader.setBool("boxes[" + i + "].material.is_metal", scene.getBox(i).getMaterial().isMetal());
            pathtraceShader.setFloat("boxes[" + i + "].material.emission", scene.getBox(i).getMaterial().getEmission());
            pathtraceShader.setFloat("boxes[" + i + "].material.roughness", scene.getBox(i).getMaterial().getRoughness());
            pathtraceShader.setBool("boxes[" + i + "].material.is_glass", scene.getBox(i).getMaterial().isGlass());
            pathtraceShader.setFloat("boxes[" + i + "].material.IOR", scene.getBox(i).getMaterial().getIOR());
            boxBounds.addBox(scene.getBox(i));
        }
        pathtraceShader.setVec3("box_bounds.min", boxBounds.min);
        pathtraceShader.setVec3("box_bounds.max", boxBounds.max);
        pathtraceShader.setVec3("triangles_offset", triangles_offset);
        pathtraceShader.setMat4("triangles_rotation", new Matrix4f().rotate(triangles_rotation));

        /*glActiveTexture(GL_TEXTURE5);
        modelTexture.bind();
        pathtraceShader.setInt("model_texture", 5);*/

        glActiveTexture(GL_TEXTURE2);
        normalTexture[prevPt].bind();
        pathtraceShader.setInt("prev_normal", 2);
        glBindImageTexture(2, normalTexture[currPt].getId(), 0, false, 0, GL_READ_WRITE, GL_RGBA32F);

        glActiveTexture(GL_TEXTURE3);
        pathtraceTexture[prevPt].bind();
        pathtraceShader.setInt("prev_color", 3);
        glBindImageTexture(3, pathtraceTexture[currPt].getId(), 0, false, 0, GL_READ_WRITE, GL_RGBA32F);

        pathtraceShader.invokeCompute();

        if (atrousFilter) {
            for (int i = 0; i < Gui.iterations[0]; i++) {
                atrousShader.use();

                atrousShader.setInt("radius", Gui.radius[0]);
                atrousShader.setInt("step_size", 1 << i);
                atrousShader.setFloat("sigma_spatial", (1 << (i + 1)) - 1 * Gui.sigma_spatial[0]);
                atrousShader.setFloat("sigma_color", 1.0f / i * Gui.sigma_color[0]);
                atrousShader.setFloat("sigma_depth", 1.0f / (1 << i) * Gui.sigma_depth[0]);
                atrousShader.setFloat("sigma_normal", 1.0f / (1 << i) * Gui.sigma_normal[0]);

                glActiveTexture(GL_TEXTURE0);
                positionTexture.bind();
                atrousShader.setInt("position_texture", 0);

                glActiveTexture(GL_TEXTURE1);
                albedoTexture.bind();
                atrousShader.setInt("albedo_texture", 1);

                glActiveTexture(GL_TEXTURE2);
                normalTexture[currPt].bind();
                atrousShader.setInt("normal_texture", 2);

                glActiveTexture(GL_TEXTURE3);
                if (i == 0) pathtraceTexture[currPt].bind();
                else atrousTexture[prevAtr].bind();
                atrousShader.setInt("color_texture", 3);

                glBindImageTexture(4, atrousTexture[currAtr].getId(), 0, false, 0, GL_READ_WRITE, GL_RGBA32F);
                atrousShader.invokeCompute();

                swapAtrous();
            }
        }
        presentShader.use();
        glActiveTexture(GL_TEXTURE1);
        albedoTexture.bind();
        presentShader.setInt("albedo_texture", 1);
        presentShader.setFloat("exposure", exposure);
        presentShader.setFloat("gamma", gamma);
        glActiveTexture(GL_TEXTURE8);
        if (atrousFilter) {
            atrousTexture[currAtr].bind();
        } else {
            pathtraceTexture[currPt].bind();
        }
        presentShader.setInt("color_texture", 8);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        Quad.draw();
        swapColorTextures();
        if (accumulation && accumulatedSamples != maxAccumulatedSamples)
            accumulatedSamples++;
        frameIndex++;
    }

    public static void resetAccFrames() {
        accumulatedSamples = 0;
    }

    private static void swapColorTextures() {
        currPt = 1 - currPt;
        prevPt = 1 - prevPt;
    }

    private static void swapAtrous() {
        currAtr = 1 - currAtr;
        prevAtr = 1 - prevAtr;
    }

    public static void resetTextures() {
        pathtraceTexture[0].bind(); pathtraceTexture[0].texImage(GL_RGBA32F, Window.getWidth(), Window.getHeight(), GL_FLOAT, 0);
        pathtraceTexture[1].bind(); pathtraceTexture[1].texImage(GL_RGBA32F, Window.getWidth(), Window.getHeight(), GL_FLOAT, 0);
        normalTexture[0].bind(); normalTexture[0].texImage(GL_RGBA32F, Window.getWidth(), Window.getHeight(), GL_FLOAT, 0);
        normalTexture[1].bind(); normalTexture[1].texImage(GL_RGBA32F, Window.getWidth(), Window.getHeight(), GL_FLOAT, 0);
        positionTexture.bind(); positionTexture.texImage(GL_RGBA32F, Window.getWidth(), Window.getHeight(), GL_FLOAT, 0);
        albedoTexture.bind(); albedoTexture.texImage(GL_RGBA32F, Window.getWidth(), Window.getHeight(), GL_FLOAT, 0);
        atrousTexture[0].bind(); atrousTexture[0].texImage(GL_RGBA32F, Window.getWidth(), Window.getHeight(), GL_FLOAT, 0);
        atrousTexture[1].bind(); atrousTexture[1].texImage(GL_RGBA32F, Window.getWidth(), Window.getHeight(), GL_FLOAT, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    /* Getters and Setters */

    public static Scene getScene() {
        return scene;
    }

    public static void setScene(Scene scene) {
        Renderer.scene = scene;
    }

    public static int getSamples() {
        return samples;
    }

    public static void setSamples(int value) {
        samples = value;
        Config.setInt("samples", value);
    }

    public static int getMaxAccumulatedSamples() {
        return maxAccumulatedSamples;
    }

    public static void setMaxAccumulatedSamples(int value) {
        resetAccFrames();
        maxAccumulatedSamples = value;
        Config.setInt("max_accumulated_samples", value);
    }

    public static int getBounces() {
        return bounces;
    }

    public static void setBounces(int value) {
        resetAccFrames();
        bounces = value;
        Config.setInt("bounces", value);
    }

    public static boolean isRussianRoulette() {
        return russianRoulette;
    }

    public static void useRussianRoulette(boolean value) {
        resetAccFrames();
        russianRoulette = value;
        Config.setBoolean("russian_roulette", value);
    }

    public static int getAccumulatedSamples() {
        return accumulatedSamples;
    }

    public static boolean isAccumulation() {
        return accumulation;
    }

    public static void useAccumulation(boolean value) {
        if (!value) resetAccFrames();
        accumulation = value;
        Config.setBoolean("accumulation", value);
    }

    public static boolean isTemporalReprojection() {
        return temporalReprojection;
    }

    public static void useTemporalReprojection(boolean value) {
        temporalReprojection = value;
        Config.setBoolean("temporal_reprojection", value);
    }

    public static float getGamma() {
        return gamma;
    }

    public static void setGamma(float value) {
        gamma = value;
        Config.setFloat("gamma", value);
    }

    public static float getExposure() {
        return exposure;
    }

    public static void setExposure(float value) {
        exposure = value;
        Config.setFloat("exposure", value);
    }

    public static boolean isTemporalAntialiasing() {
        return temporalAntialiasing;
    }

    public static void useTemporalAntialiasing(boolean value) {
        resetAccFrames();
        temporalAntialiasing = value;
        Config.setBoolean("temporal_antialiasing", value);
    }
    public static boolean ischeckerboardRendering() {
        return checkerboardRendering;
    }

    public static void usecheckerboardRendering(boolean value) {
        resetAccFrames();
        checkerboardRendering = value;
        Config.setBoolean("checkerboard_rendering", value);
    }

    public static boolean isAtrousFilter() {
        return atrousFilter;
    }

    public static void useAtrousFilter(boolean value) {
        atrousFilter = value;
        Config.setBoolean("atrous_filter", value);
    }

    public static float getFocusDistance() {
        return focusDistance;
    }

    public static void setFocusDistance(float value) {
        resetAccFrames();
        focusDistance = value;
        Config.setFloat("focus_distance", value);
    }

    public static float getAperture() {
        return aperture;
    }

    public static void setAperture(float value) {
        resetAccFrames();
        aperture = value;
        Config.setFloat("aperture", value);
    }

    public static float getFOV() {
        return fov;
    }

    public static void setFOV(float value) {
        resetAccFrames();
        fov = value;
        Config.setFloat("fov", value);
    }
}