#include <chrono>
#include <random>
#include "raylib.h"
#include <iostream>
#include <vector>
#include <future>
#include <omp.h>
#include <array>
#include <algorithm>
#include <unordered_set>
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

//---------------------------------------------------------------------------------------------------------------------------------

std::mt19937 CreateGeneratorWithTimeSeed();
float RandomFloat(float min, float max, std::mt19937& rng);

//---------------------------------------------------------------------------------------------------------------------------------


const int screenWidth = 2560, screenHeight = 1440, numThreads = 24;
int startingNumParticles = 20000, startingClusterParticles = 1;
const float collisionThreshold = 3.5f;
Vector2 particleSize = {2,2};

std::mt19937 rng = CreateGeneratorWithTimeSeed();

std::mutex particlesMutex;

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
    const float halfWidth = boundary.width/2.0f, halfHeight = boundary.height/2.0f;

    //std::vector<Particle*> particles;
    std::vector<Particle> particles;
    std::array<std::unique_ptr<QuadTree>, 4> children {nullptr,nullptr,nullptr,nullptr};   //quadrants labeled as in unit circle

    QuadTree(Rectangle boundary)
        :boundary(boundary), children{nullptr,nullptr,nullptr,nullptr}
    {}

    void InsertAllParticles(std::vector<Particle>& allParticles){

    }

    bool isLeaf(){
        if(children[0] == nullptr){
            return true;
        }
        return false;
    }

    void debugPrint(){
        int count = 0;
        for(int i = 0 ; i < 4; i++){
            if(children[i]){
                count++;
            }
        }
        std::cout << "# stored particles: " << particles.size() << "\ton level: " << treeDepth << "\tand " << count << " non-null children" <<std::endl;
        for(int i = 0 ; i < 4; i++){
            if(children[i]){
                children[i]->debugPrint();
            }
        }
    }

    void Insert(Particle& p) {
        bool isCollision = CheckCollisionPointRec(p.pos, boundary);
        if (!isCollision) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(particlesMutex); // Lock the mutex
            if (particles.size() < capacity) {
                particles.push_back(p);        
                return;
            }
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
            {
                std::lock_guard<std::mutex> lock(particlesMutex); // Lock the mutex
                particles.push_back(p);    
            }
        }
    }

    std::vector<Particle> queryCircle(const Vector2& center, float radius) {
        std::vector<Particle> results;

        // Check if the boundary intersects the circle
        if (!CheckCollisionCircleRec(center, radius, boundary)) {
            return results;
        }

        // Check each particle in this node
        for (auto& p : particles) {
            if (CheckCollisionPointCircle(p.pos, center, radius)) {
                results.push_back(p);
            }
        }

        // Recursively search each child node
        if (isDivided) {
            for (auto& child : children) {
                auto childResults = child->queryCircle(center, radius);
                results.insert(results.end(), childResults.begin(), childResults.end());
            }
        }

        return results;
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

    void clear(){
        particles.clear();
        for(int i = 0; i < 4; i++){
            if(children[i]){
                children[i]->clear();
            }
        }
    }

    std::chrono::duration<float> Draw() {
        using namespace std::literals::chrono_literals;

        auto start = std::chrono::high_resolution_clock::now();

        for (auto& p : particles) {
            DrawPixelV(p.pos, p.color);
        }

        //DrawRectangleLinesEx(boundary, 1, GREEN);

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

    void eraseParticle(int j){
        particles.erase(particles.begin() + j);
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

std::vector<Particle> randomWalkMultiThreaded(std::vector<Particle> FreeParticles){
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
                newParticle.RandomWalk(1, 1);
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

        return newParticles;
}


std::vector<Particle> checkCollisionsSingleThread(std::vector<Particle>& cluster, std::vector<Particle>& FreeParticles, QuadTree& qt){
        std::vector<Particle> newParticles, result;

        for(auto it = cluster.begin(); it != cluster.end(); ++it){
            newParticles = {qt.queryCircle(it->pos, collisionThreshold)};
            for(auto& p : newParticles){
                if(CheckCollisionPointCircle(it->pos, p.pos, collisionThreshold / 1.5f)){
                    result.push_back(p);
                    for(auto zt = FreeParticles.begin(); zt != FreeParticles.end(); ++zt){
                        if(zt->pos.x == p.pos.x and zt->pos.y == p.pos.y){
                            FreeParticles.erase(zt);
                            break;  // exit the inner loop after erasing the particle
                        }
                    }
                }
            }
        }
        return result;   
}


std::vector<Particle> checkCollisions(std::vector<Particle>& cluster, std::vector<Particle>& FreeParticles, QuadTree& qt){
    if(false){
        // Split the cluster vector into smaller chunks
        int chunkSize = cluster.size() / numThreads;
        std::vector<std::vector<Particle>> particleChunks(numThreads);

        for (int i = 0; i < numThreads; i++) {
            int startIndex = i * chunkSize;
            int endIndex = (i == numThreads - 1) ? cluster.size() : (i + 1) * chunkSize;
            particleChunks[i] = std::vector<Particle>(cluster.begin() + startIndex, cluster.begin() + endIndex);
        }

        for(int i = 0; i < numThreads; i++){
            //std::async(std::launch::async, checkCollisionsSingleThread(particleChunks[i], FreeParticles, qt));
        }


    }
    else{
        return checkCollisionsSingleThread(cluster, FreeParticles, qt);  
    }
}


//---------------------------------------------------------------------------------------------------------------------------------

int main() {

    float queryRadius = 400;

    InitWindow(screenWidth, screenHeight, "DLA");
    SetTargetFPS(100);
    omp_set_num_threads(20);

    //std::vector<Particle> FreeParticles(startingNumParticles,Particle(1000,700,RED));
    std::vector<Particle> FreeParticles = CreateCircle(20000,RED,{screenWidth/2.0,screenHeight/2.0}, queryRadius);
    //std::vector<Particle> FreeParticles(1000, Particle(RandomFloat(0,screenWidth, rng),RandomFloat(0,screenHeight, rng), RED));     //random particles
    std::vector<Particle> ClusterParticles(startingClusterParticles,Particle(screenWidth/2.0,screenHeight/2.0,GREEN));

    Camera2D camera = { 0 };
    camera.target = { screenWidth / 2.0f, screenHeight / 2.0f };
    camera.offset = { screenWidth / 2.0f, screenHeight / 2.0f };
    camera.rotation = 0.0f;
    camera.zoom = 1.0f;

    QuadTree qt(Rectangle{0,0,screenWidth,screenHeight});

    //main loop
    for (int i = 0; !WindowShouldClose(); i++) {

        FreeParticles = randomWalkMultiThreaded(FreeParticles);

        //clear quadtree
        qt.clear();

        //rebuilt quadtree
        for(long long unsigned int i = 0; i < FreeParticles.size(); i++){
            qt.Insert(FreeParticles[i]);
        }

        //check for points in central circle using qt
        //std::vector<Particle> centralPoints {qt.queryCircle({screenWidth/2.0,screenHeight/2.0}, queryRadius)};

        //make centralPoints green
        /*for(unsigned int i = 0; i < centralPoints.size(); i++){
            Particle p = centralPoints[i];
            p.color = GREEN;
            centralPoints[i] = p; 
        }*/


        std::vector<Particle> collided {checkCollisions(ClusterParticles, FreeParticles, qt)};

        for(auto p: collided){
            p.color = PURPLE;
            ClusterParticles.push_back(p);
        }


        BeginDrawing();

        ClearBackground(BLACK);
        DrawFPS(20, 20);

        //draw cluster
        for (long long unsigned int i = 0; i < ClusterParticles.size(); i++) {
            //DrawRectangleV(ClusterParticles[i].pos, { 2,2 }, ClusterParticles[i].color);
            DrawPixelV(ClusterParticles[i].pos,ClusterParticles[i].color);
        }

        //draw quadtree + freepoints
        std::chrono::duration<float> drawDuration = qt.Draw();

        //draw center circle
        //DrawCircleLines(screenWidth/2.0, screenHeight/2.0, queryRadius, ORANGE);

        //draw centralPoints
        /*for(long long unsigned int i = 0; i < centralPoints.size(); i++){
            DrawPixelV(centralPoints[i].pos,centralPoints[i].color);
        }*/

        EndDrawing();

        if(i % 2000 == 0){
            //qt.debugPrint();
        }

    }

    CloseWindow();

    return 0;
}
