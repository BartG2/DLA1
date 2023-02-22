#include <chrono>
#include <random>
#include "raylib.h"
#include <iostream>
#include <vector>
#include <future>

//---------------------------------------------------------------------------------------------------------------------------------

std::mt19937 CreateGeneratorWithTimeSeed();
float RandomFloat(float min, float max, std::mt19937& rng);

//---------------------------------------------------------------------------------------------------------------------------------


const int screenWidth = 2560, screenHeight = 1440, screenDepth = 2000, numThreads = 20;
const int startingNumParticles = 200, startingAggregateParticles = 1, particleSize = 20;
Vector3 particleSizeVector = { particleSize,particleSize,particleSize};
float collisionDistance = 2.0, cameraMoveSpeed = 2.0f;

std::mt19937 rng = CreateGeneratorWithTimeSeed();

//---------------------------------------------------------------------------------------------------------------------------------

class Particle {
public:
    Vector3 pos;
    Color color;
    bool isStuck;

    Particle(float x, float y, float z, Color col) {
        pos = { x, y, z };
        color = col;
        isStuck = false;
    }

    Particle(Vector3 position, Color col) {
        pos = position;
        color = col;
        isStuck = false;
    }

    Particle() {
        pos = { screenWidth - 10, screenHeight - 10, 0 };
        color = WHITE;
        isStuck = false;
    }

    void RandomWalk(float stepSize, int numSteps) {
        for (int i = 0; i < numSteps; i++) {
            float dx = RandomFloat(-1, 1, rng);
            float dy = RandomFloat(-1, 1, rng);
            float dz = RandomFloat(-1, 1, rng);

            float newX = pos.x + dx * stepSize;
            float newY = pos.y + dy * stepSize;
            float newZ = pos.z + dz * stepSize;

            // Check if particle is out of bounds and correct position
            if (newX < 0) {
                newX = 0;
            }
            else if (newX > screenWidth) {
                newX = screenWidth;
            }
            if (newY < 0) {
                newY = 0;
            }
            else if (newY > screenHeight) {
                newY = screenHeight;
            }
            if (newZ < 0) {
                newZ = 0;
            }
            else if (newZ > screenDepth) {
                newZ = screenDepth;
            }

            pos = { newX, newY, newZ };
        }
    }
};


//---------------------------------------------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------------------------------------------

std::mt19937 CreateGeneratorWithTimeSeed() {
    // Get the current time in nanoseconds
    auto now = std::chrono::high_resolution_clock::now();
    auto nanos = std::chrono::time_point_cast<std::chrono::nanoseconds>(now).time_since_epoch().count();

    // Create a new mt19937 generator and seed it with the current time in nanoseconds
    std::mt19937 gen(static_cast<unsigned int>(nanos));
    return gen;
}

float RandomFloat(float min, float max, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

float Vector3Distance(const Vector3& a, const Vector3& b) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}




//---------------------------------------------------------------------------------------------------------------------------------

int main() {

    InitWindow(screenWidth, screenHeight, "DLA");
    SetTargetFPS(100);

    //std::vector<Particle> FreeParticles(startingNumParticles,Particle(500,500,RED));
    //std::vector<Particle> AggregateParticles(startingAggregateParticles,Particle(screenWidth/2.0,screenHeight/2.0,WHITE));

    std::vector<Particle> FreeParticles(startingNumParticles, Particle({screenWidth/3.0,screenHeight/3.0,screenDepth/3.0}, RED));
    std::vector<Particle> AggregateParticles(startingAggregateParticles, Particle({ screenWidth / 2.0, screenHeight / 2.0, screenDepth/3.0 }, WHITE));

    /*Camera2D camera = {0};
    camera.target = { screenWidth / 2.0f, screenHeight / 2.0f };
    camera.offset = { screenWidth / 2.0f, screenHeight / 2.0f };
    camera.rotation = 0.0f;
    camera.zoom = 1.0f;*/


    Camera3D camera = { 0 };
    camera.position = { 0.0f, 0.0f, 10.0f };
    camera.target = { screenWidth/2.0,screenHeight/2.0,screenDepth/2.0};
    camera.up = { 0.0f, 1.0f, 0.0f };
    camera.fovy = 100.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    Vector2 lastMousePos = GetMousePosition();

    //SetCameraMode(camera, CAMERA_FIRST_PERSON);
    //SetCameraMoveControls(KEY_W, KEY_S, KEY_D, KEY_A, KEY_UP,KEY_DOWN);

    //main loop
    while (!WindowShouldClose()) {

        // Check arrow keys input and move camera accordingly
        if (IsKeyDown(KEY_RIGHT)) {
            camera.position.z += cameraMoveSpeed;
        }
        if (IsKeyDown(KEY_LEFT)) {
            camera.position.z -= cameraMoveSpeed;
        }
        if (IsKeyDown(KEY_UP)) {
            camera.position.y += cameraMoveSpeed;
        }
        if (IsKeyDown(KEY_DOWN)) {
            camera.position.y -= cameraMoveSpeed;
        }
        if (IsKeyDown(KEY_W)) {
            camera.position.x += cameraMoveSpeed;
        }
        if (IsKeyDown(KEY_S)) {
            camera.position.x -= cameraMoveSpeed;
        }

        /*// Panning camera with mouse
        Vector2 currentMousePos = GetMousePosition();
        float mouseDeltaX = currentMousePos.x - lastMousePos.x;
        float mouseDeltaY = currentMousePos.y - lastMousePos.y;
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            camera.target.x -= mouseDeltaX * cameraMoveSpeed;
            camera.target.y += mouseDeltaY * cameraMoveSpeed;
        }*/

        float panSpeed = 5.0f;
        Vector2 newMousePos = GetMousePosition();
        Vector2 mouseDelta = { newMousePos.x - lastMousePos.x, newMousePos.y - lastMousePos.y };
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) || IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
            camera.target.x -= mouseDelta.x * panSpeed;
            camera.target.y -= mouseDelta.y * panSpeed;
        }
        lastMousePos = newMousePos;


        for (int i = 0; i < FreeParticles.size(); i++) {
            FreeParticles[i].RandomWalk(1,2);
        }

        for (int i = 0; i < AggregateParticles.size(); i++) {
            //AggregateParticles[i].RandomWalk(1, 1);
        }



        // Split the FreeParticles vector into smaller chunks
        int chunkSize = FreeParticles.size() / numThreads;
        std::vector<std::vector<Particle>> particleChunks(numThreads);
        for (int i = 0; i < numThreads; i++) {
            int startIndex = i * chunkSize;
            int endIndex = (i == numThreads - 1) ? FreeParticles.size() : (i + 1) * chunkSize;
            particleChunks[i] = std::vector<Particle>(FreeParticles.begin() + startIndex, FreeParticles.begin() + endIndex);
        }

        // Create a vector of futures to hold the results of each thread
        std::vector<std::future<std::vector<Particle>>> futures(numThreads);

        // Launch each thread to calculate the movement of its subset of particles
        for (int i = 0; i < numThreads; i++) {
            futures[i] = std::async(std::launch::async, [](const std::vector<Particle>& particles) {
                std::vector<Particle> newParticles;
            newParticles.reserve(particles.size());
            for (const auto& particle : particles) {
                Particle newParticle = particle;
                newParticle.RandomWalk(1, 2);
                newParticles.push_back(newParticle);
            }
            return newParticles;
                }, particleChunks[i]);
        }

        // Wait for each thread to finish and combine the results into a single vector
        std::vector<Particle> newParticles;
        for (int i = 0; i < numThreads; i++) {
            std::vector<Particle> result = futures[i].get();
            newParticles.insert(newParticles.end(), result.begin(), result.end());
        }

        // Replace the old FreeParticles vector with the new one
        FreeParticles = newParticles;

        BeginDrawing();

        ClearBackground(BLACK);
        DrawFPS(20, 20);

        /*BeginMode2D(camera);
        for (int i = 0; i < FreeParticles.size(); i++) {
            DrawRectangleV(FreeParticles[i].pos, { 2,2 }, FreeParticles[i].color);
        }

        for (int i = 0; i < AggregateParticles.size(); i++) {
            DrawRectangleV(AggregateParticles[i].pos, { 2,2 }, AggregateParticles[i].color);
        }
        EndMode2D();*/

        BeginMode3D(camera);

        DrawPlane({ 0,0,0 }, { 3000,3000 }, PURPLE);
        DrawPlane({ 0,1000,0 }, { 3000,3000 }, GREEN);
        DrawCubeWiresV({ 0,0,0 }, { screenWidth,screenHeight,screenDepth }, ORANGE);

        for (int i = 0; i < AggregateParticles.size(); i++) {
            DrawCubeV(AggregateParticles[i].pos, particleSizeVector, AggregateParticles[i].color);
        }

        for (int i = 0; i < FreeParticles.size(); i++) {
            DrawCubeV(FreeParticles[i].pos, particleSizeVector, FreeParticles[i].color);
        }

        EndMode3D();


        EndDrawing();
    }

    CloseWindow();

    return 0;
}
