#include <chrono>
#include <random>
#include "raylib.h"
#include <iostream>
#include <vector>
#include <future>
#include <omp.h>
#include <array>
#include <mutex>
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

//---------------------------------------------------------------------------------------------------------------------------------

std::mt19937 CreateGeneratorWithTimeSeed();
float RandomFloat(float min, float max, std::mt19937& rng);

//---------------------------------------------------------------------------------------------------------------------------------


const int screenWidth = 2560, screenHeight = 1440, numThreads = 20;
int startingNumParticles = 20000, startingClusterParticles = 1;
const float collisionThreshold = 3.5f;
Vector2 particleSize = {2,2};

std::mt19937 rng = CreateGeneratorWithTimeSeed();

//---------------------------------------------------------------------------------------------------------------------------------

class Particle {
public:
    Vector2 pos;
    Color color;
    bool isStuck;

    Particle(float x, float y, Color col)
        :pos({x,y}), color(col)
    {}

    Particle()
        :pos({ screenWidth - 10, screenHeight - 10 }), color(WHITE), isStuck(false)
    {}

    void RandomWalk(float stepSize, int numSteps) {
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

};

class QuadTree {
public:
    Rectangle boundary;
    const static int capacity = 40;
    const static int maxTreeDepth = 6;
    int treeDepth = 0;
    bool isDivided = false;
    //std::vector<Particle*> particles;
    std::vector<std::reference_wrapper<Particle>> particles;
    std::array<std::unique_ptr<QuadTree>, 4> children;   //quadrants labeled as in unit circle

    std::mutex particlesMutex;

    float halfWidth = boundary.width/2.0f, halfHeight = boundary.height/2.0f;

    QuadTree(Rectangle boundary)
        :boundary(boundary)
    {}

    void Insert(Particle& p) {
        bool isCollision = CheckCollisionPointRec(p.pos, boundary);
        if (!isCollision) {
            return;
        }

        std::lock_guard<std::mutex> lock(particlesMutex);  // acquire mutex lock

        if (particles.size() < capacity) {
            particles.push_back(p);
            return;
        }

        if (treeDepth < maxTreeDepth) {
            if (!isDivided) {
                Divide();
                isDivided = true;
            }

            children[0]->Insert(p);
            children[1]->Insert(p);
            children[2]->Insert(p);
            children[3]->Insert(p);
        }
        else {
            particles.push_back(p);
        }
    }

    void InsertBatch(std::vector<Particle>& particles) {
    for (Particle& p : particles) {
        Insert(p);
        }
    }

    void InsertParticles(std::vector<Particle>& particles) {
        const int batchSize = 1000;
        const int numBatches = (particles.size() + batchSize - 1) / batchSize;

        std::vector<std::future<void>> futures;
        futures.reserve(numBatches);

        for (int i = 0; i < numBatches; i++) {
            int startIndex = i * batchSize;
            int endIndex = std::min(startIndex + batchSize, static_cast<int>(particles.size()));

            std::vector<Particle> batch(particles.begin() + startIndex, particles.begin() + endIndex);

            futures.push_back(std::async(std::launch::async, &InsertBatch, std::move(batch)));
        }

        for (auto& future : futures) {
            future.get();
        }
    }


    void Divide() {
        //in order of unit circle quadrants
        children[0] = std::make_unique<QuadTree>(Rectangle{ boundary.x + halfWidth, boundary.y, halfWidth, halfHeight });
        children[1] = std::make_unique<QuadTree>(Rectangle{ boundary.x, boundary.y, halfWidth, halfHeight });
        children[2] = std::make_unique<QuadTree>(Rectangle{ boundary.x, boundary.y + halfHeight, halfWidth, halfHeight });
        children[3] = std::make_unique<QuadTree>(Rectangle{ boundary.x + halfWidth, boundary.y + halfHeight, halfWidth, halfHeight });

        for (auto& p : particles) {
            for (auto& child : children) {
                child->Insert(p);
            }
        }

        particles.clear();

        for (auto& child : children) {
            child->treeDepth = treeDepth + 1;
        }
    }

    std::chrono::duration<float> Draw() {
        using namespace std::literals::chrono_literals;

        auto start = std::chrono::high_resolution_clock::now();

        for (auto& p : particles) {
            DrawPixelV(p.get().pos, p.get().color);
        }

        DrawRectangleLinesEx(boundary, 1, GREEN);

        if (isDivided) {
            for (auto& child : children) {
                if(child->particles.size() > 0){
                    child->Draw();
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;
        return duration;
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

float vector2distance(Vector2 v1, Vector2 v2) {
    float dx = v2.x - v1.x;
    float dy = v2.y - v1.y;
    return std::sqrt(dx * dx + dy * dy);
}

/*__global__ void vector2distanceKernel(Vector2* v1, Vector2* v2, float* result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float dx = v2[idx].x - v1[idx].x;
    float dy = v2[idx].y - v1[idx].y;
    result[idx] = sqrtf(dx * dx + dy * dy);
}*/

/*void vector2distanceCUDA(std::vector<Vector2>& v1, std::vector<Vector2>& v2, std::vector<float>& result)
{
    const int size = v1.size();
    Vector2* d_v1;
    Vector2* d_v2;
    float* d_result;

    cudaMalloc((void**)&d_v1, size * sizeof(Vector2));
    cudaMalloc((void**)&d_v2, size * sizeof(Vector2));
    cudaMalloc((void**)&d_result, size * sizeof(float));

    cudaMemcpy(d_v1, v1.data(), size * sizeof(Vector2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, v2.data(), size * sizeof(Vector2), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    vector2distanceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_v1, d_v2, d_result);

    result.resize(size);
    cudaMemcpy(result.data(), d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_result);
}*/

std::vector<Particle> CreateCircle(int numParticles, Color col, Vector2 center, float radius){
    float degreeIncrement = 360.0f/(float)numParticles;
    std::vector<Particle> particles;

    for(float i = 0; i < 360; i += degreeIncrement){
        float x = radius * cos(i) + center.x;
        float y = radius * sin(i) + center.y;
        Particle p(x,y,col);
        particles.push_back(p);
    }

    return particles;
}


//---------------------------------------------------------------------------------------------------------------------------------

int main() {

    InitWindow(screenWidth, screenHeight, "DLA");
    SetTargetFPS(100);
    omp_set_num_threads(20);

    //std::vector<Particle> FreeParticles(startingNumParticles,Particle(1000,700,RED));
    std::vector<Particle> FreeParticles = CreateCircle(100000,RED,{screenWidth/2.0,screenHeight/2.0}, 500);
    //std::vector<Particle> FreeParticles(1000, Particle(RandomFloat(0,screenWidth, rng),RandomFloat(0,screenHeight, rng), RED));     //random particles
    std::vector<Particle> ClusterParticles(startingClusterParticles,Particle(screenWidth/2.0,screenHeight/2.0,WHITE));

    Camera2D camera = { 0 };
    camera.target = { screenWidth / 2.0f, screenHeight / 2.0f };
    camera.offset = { screenWidth / 2.0f, screenHeight / 2.0f };
    camera.rotation = 0.0f;
    camera.zoom = 1.0f;

    //main loop
    while (!WindowShouldClose()) {

        auto startMainLoop = std::chrono::high_resolution_clock::now();

        /*#pragma omp parallel for
        for(long long unsigned int i = 0; i < FreeParticles.size(); i++){
            FreeParticles[i].RandomWalk(1,2);
        }*/

        /*#pragma omp parallel for
        for (long long unsigned int i = 0; i < ClusterParticles.size(); i++){
            for(long long unsigned int j = 0; j < FreeParticles.size(); j++){
                float distance = vector2distance(ClusterParticles[i].pos, FreeParticles[j].pos);

                if (distance < collisionThreshold) {
                    FreeParticles[j].isStuck = true;
                    FreeParticles[j].color = WHITE;
                    ClusterParticles.push_back(FreeParticles[j]);
                    FreeParticles.erase(FreeParticles.begin() + j);
                    break;
                }
            }
        }*/

        auto startVectorSplit = std::chrono::high_resolution_clock::now();
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

        auto endVectorSplit = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> vectorSplitDuration = endVectorSplit - startVectorSplit;

        auto startVectorCopy = std::chrono::high_resolution_clock::now();
        // Replace the old FreeParticles vector with the new one
        FreeParticles = newParticles;
        auto endVectorCopy = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> vectorCopyDuration = endVectorCopy - startVectorCopy;

        auto startQT = std::chrono::high_resolution_clock::now();
        QuadTree qt(Rectangle{0,0,screenWidth,screenHeight});
        auto endQT = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> qtDuration = endQT - startQT;

        auto start = std::chrono::high_resolution_clock::now();
        for(long long unsigned int i = 0; i < FreeParticles.size(); i++){
            //const Particle& p = FreeParticles[i];
            //Particle *p = &FreeParticles[i];
            qt.Insert(FreeParticles[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> insertDuration = end - start;


        BeginDrawing();

        ClearBackground(BLACK);
        DrawFPS(20, 20);

        /*//#pragma omp parallel for
        for (long long unsigned int i = 0; i < FreeParticles.size(); i++) {
            DrawRectangleV(FreeParticles[i].pos, { 2,2 }, FreeParticles[i].color);
        }*/
        //#pragma omp parallel for
        for (long long unsigned int i = 0; i < ClusterParticles.size(); i++) {
            //DrawRectangleV(ClusterParticles[i].pos, { 2,2 }, ClusterParticles[i].color);
            DrawPixelV(ClusterParticles[i].pos,ClusterParticles[i].color);
        }
        std::chrono::duration<float> drawDuration = qt.Draw();



        EndDrawing();

        auto endMainLoop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> mainLoopDuration = endMainLoop - startMainLoop;

        std::cout << "Main Loop: " << mainLoopDuration.count() << "\tInsert: " << insertDuration.count()/mainLoopDuration.count() << "\tDraw: " << drawDuration.count()/mainLoopDuration.count() << "\tQT: " << std::fixed << qtDuration.count()/mainLoopDuration.count() << "\tVectorCopy: " << vectorCopyDuration.count()/mainLoopDuration.count() << "\tVectorSplit: " << vectorSplitDuration.count()/mainLoopDuration.count() << "\tRemainder: " << 1.0 - vectorSplitDuration.count()/mainLoopDuration.count() - vectorCopyDuration.count()/mainLoopDuration.count() - qtDuration.count()/mainLoopDuration.count() - drawDuration.count()/mainLoopDuration.count() - insertDuration.count()/mainLoopDuration.count() << std::endl;
    }

    CloseWindow();

    return 0;
}
