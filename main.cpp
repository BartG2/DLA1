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
int startingNumParticles = 2000, startingClusterParticles = 1;
float collisionDistance = 2.0;

std::mt19937 rng = CreateGeneratorWithTimeSeed();

//---------------------------------------------------------------------------------------------------------------------------------

class Particle {
public:
    Vector2 pos;
    Vector3 pos3D;
    Color color;
    bool isStuck, is3D;

    Particle(float x, float y, Color col) {
        pos = { x,y };
        pos3D = { x,y,0 };
        color = col;
        isStuck = false;
        is3D = false;
    }
    Particle(Vector2 position, Color col) {
        pos = position;
        color = col;
        isStuck = false;
        is3D = false;
    }

    Particle(Vector3 position, Color col) {
        pos3D = position;
        color = col;
        isStuck = false;
        is3D = true;
    }

    Particle() {
        pos = { screenWidth - 10, screenHeight - 10 };
        pos3D = { screenWidth - 10, screenHeight - 10, 0 };
        color = WHITE;
        isStuck = false;
    }

    void RandomWalk(float stepSize, int numSteps) {
        if (!is3D) {
            for (int i = 0; i < numSteps; i++) {
                float dx = RandomFloat(-1, 1, rng);
                float dy = RandomFloat(-1, 1, rng);

                float newX = pos.x + dx * stepSize;
                float newY = pos.y + dy * stepSize;



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

                pos.x = newX;
                pos.y = newY;
            }
        }
        else if (is3D) {
            for (int i = 0; i < numSteps; i++) {
                float dx = RandomFloat(-1, 1, rng);
                float dy = RandomFloat(-1, 1, rng);
                float dz = RandomFloat(-1, 1, rng);

                float newX = pos3D.x + dx * stepSize;
                float newY = pos3D.y + dy * stepSize;
                float newZ = pos3D.z + dz * stepSize;



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

                pos3D = { newX,newY,newZ };
            }
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

    std::vector<Particle> FreeParticles(startingNumParticles,Particle(500,500,RED));
    std::vector<Particle> ClusterParticles(startingClusterParticles,Particle(screenWidth/2.0,screenHeight/2.0,WHITE));

    Camera2D camera = { 0 };
    camera.target = { screenWidth / 2.0f, screenHeight / 2.0f };
    camera.offset = { screenWidth / 2.0f, screenHeight / 2.0f };
    camera.rotation = 0.0f;
    camera.zoom = 1.0f;

    //main loop
    while (!WindowShouldClose()) {

        for (int i = 0; i < FreeParticles.size(); i++) {
            FreeParticles[i].RandomWalk(1,2);
        }

        for (int i = 0; i < ClusterParticles.size(); i++) {
            //ClusterParticles[i].RandomWalk(1, 1);
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

        BeginMode2D(camera);
        for (int i = 0; i < FreeParticles.size(); i++) {
            DrawRectangleV(FreeParticles[i].pos, { 2,2 }, FreeParticles[i].color);
            //DrawPixelV(FreeParticles[i].pos, FreeParticles[i].color);
            //DrawCircleV(FreeParticles[i].pos, 2, FreeParticles[i].color);
        }

        for (int i = 0; i < ClusterParticles.size(); i++) {
            //DrawCircleV(ClusterParticles[i].pos, 3, ClusterParticles[i].color);
            DrawRectangleV(ClusterParticles[i].pos, { 2,2 }, ClusterParticles[i].color);
        }
        EndMode2D();


        EndDrawing();
    }

    CloseWindow();

    return 0;
}
