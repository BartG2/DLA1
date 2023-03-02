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
#include <thread>
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

//---------------------------------------------------------------------------------------------------------------------------------

std::mt19937 CreateGeneratorWithTimeSeed();
float RandomFloat(float min, float max, std::mt19937& rng);

//---------------------------------------------------------------------------------------------------------------------------------


const int screenWidth = 2560, screenHeight = 1440, numThreads = 2;
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

class Timer{
public:
    std::chrono::time_point<std::chrono::high_resolution_clock> startPoint;

    Timer(){
        startPoint = std::chrono::high_resolution_clock::now();
    }

    ~Timer(){
        stop();
    }

    void stop(){
        auto endPoint = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(startPoint).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endPoint).time_since_epoch().count();

        auto duration = end - start;
        double ms = duration * 0.001;

        std::cout << "ms: " << ms << std::endl;
    }
};

class QuadTree {
public:
    Rectangle boundary;
    const static int capacity = 40;
    const static int maxTreeDepth = 6;
    int treeDepth = 0;
    bool isDivided = false;
    float halfWidth;
    float halfHeight;

    std::mutex particlesMutex; // add mutex member variable
    std::mutex isDividedMutex;
    std::mutex childrenMutex;

    //std::vector<Particle*> particles;
    std::vector<Particle> particles;
    std::array<std::unique_ptr<QuadTree>, 4> children {nullptr,nullptr,nullptr,nullptr};   //quadrants labeled as in unit circle

    QuadTree(Rectangle boundary)
        :boundary(boundary), children{nullptr,nullptr,nullptr,nullptr}
    {
        halfWidth = boundary.width/2.0f;
        halfHeight = boundary.height/2.0f;
    }

    ~QuadTree() {
        for (auto& child : children) {
            if (child) {
                delete child.release();
            }
        }
    }

    bool isLeaf(){
        if(children[0] == nullptr){
            return true;
        }
        return false;
    }

    void InsertAllParticlesParallel(std::vector<Particle>& allParticles) {
        Timer t;
        const int numChunks = numThreads;
        const int chunkSize = allParticles.size() / numChunks;
        std::vector<std::future<void>> futures;
        for (int i = 0; i < numChunks; ++i) {
            auto begin = allParticles.begin() + i * chunkSize;
            auto end = (i == numChunks - 1) ? allParticles.end() : begin + chunkSize;
            futures.emplace_back(std::async(std::launch::async, [this](auto begin, auto end) {
                for (auto it = begin; it != end; ++it) {
                    this->Insert(*it);
                }
            }, begin, end));
        }
        for (auto& future : futures) {
            future.wait();
        }
    }

    /*void InsertAllParticlesParallel(std::vector<Particle>& allParticles) {
        Timer t;
        const int numChunks = numThreads;
        const int chunkSize = allParticles.size() / numChunks;

        // create a thread pool with numThreads worker threads
        ThreadPool pool(numThreads);

        // submit tasks to the thread pool
        for (int i = 0; i < numChunks; ++i) {
            auto begin = allParticles.begin() + i * chunkSize;
            auto end = (i == numChunks - 1) ? allParticles.end() : begin + chunkSize;
            pool.submit([this, begin, end]() {
                for (auto it = begin; it != end; ++it) {
                    this->Insert(*it);
                }
            });
        }

        // wait for all tasks to finish
        pool.waitAll();
    }*/


    void InsertAllParticles(std::vector<Particle>& allParticles) {
        for(auto& p : allParticles){
            Insert(p);
        }
    }

    void Insert(const Particle& p) {
        bool isCollision = CheckCollisionPointRec(p.pos, boundary);
        if (!isCollision) {
            return;
        }

        std::lock_guard<std::mutex> lock(particlesMutex);

        if (particles.size() < capacity) {
            if (particles.size() < capacity) {
                particles.push_back(std::move(p));
            }
            return;
        }

        if (treeDepth < maxTreeDepth) {
            if (!isDivided) {
                std::lock_guard<std::mutex> lock(isDividedMutex);
                Divide();
                isDivided = true;
            }

            std::lock_guard<std::mutex> childrenLock(childrenMutex);
            children[0]->Insert(p);
            children[1]->Insert(p);
            children[2]->Insert(p);
            children[3]->Insert(p);
        }
        else {
            particles.push_back(std::move(p));
        }
    }

    std::vector<Particle> checkCollisions(std::vector<Particle>& cluster){
        std::vector<Particle> collided;

        for(unsigned int i = 0; i < cluster.size(); i++){
            std::vector<Particle> inRange = {queryCircle(cluster[i].pos, collisionThreshold)};

            for(unsigned int j = 0; j < inRange.size(); j++){
                if(CheckCollisionPointCircle(inRange[j].pos, cluster[i].pos, collisionThreshold)){
                    inRange[j].isStuck = true;
                    collided.push_back(inRange[j]);
                    removeParticle(inRange[j]);
                }
            }
        }

        return collided;
    }

    std::vector<Particle> queryCircle(const Vector2& center, float radius) {
        std::vector<Particle> results;

        std::lock_guard<std::mutex> lock(particlesMutex); // Lock the mutex

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

    /*std::vector<Particle> queryCircle(const Vector2& center, float radius) {
        std::vector<Particle> results;
        std::lock_guard<std::mutex> lock(particlesMutex); // Lock the mutex
        // Check if the boundary intersects the circle
        if (!CheckCollisionCircleRec(center, radius, boundary)) {
            return results;
        }
        // Divide particles into chunks
        const int numChunks = 2;
        const int chunkSize = particles.size() / numChunks;
        std::vector<std::future<std::vector<Particle>>> futures;
        for (int i = 0; i < numChunks; ++i) {
            auto begin = particles.begin() + i * chunkSize;
            auto end = (i == numChunks - 1) ? particles.end() : begin + chunkSize;
            futures.emplace_back(std::async(std::launch::async, [center, radius](auto begin, auto end) {
                std::vector<Particle> chunkResults;
                for (auto it = begin; it != end; ++it) {
                    if (CheckCollisionPointCircle(it->pos, center, radius)) {
                        chunkResults.push_back(*it);
                    }
                }
                return chunkResults;
            }, begin, end));
        }
        // Combine results from all chunks
        for (auto& future : futures) {
            auto chunkResults = future.get();
            results.insert(results.end(), chunkResults.begin(), chunkResults.end());
        }
        // Recursively search each child node
        if (isDivided) {
            for (auto& child : children) {
                auto childResults = child->queryCircle(center, radius);
                results.insert(results.end(), childResults.begin(), childResults.end());
            }
        }
        return results;
    }*/

    void Divide() {
        //in order of unit circle quadrants
        std::lock_guard<std::mutex> lock(childrenMutex);
        children[0] = std::make_unique<QuadTree>(Rectangle{ boundary.x + halfWidth, boundary.y, halfWidth, halfHeight });
        children[1] = std::make_unique<QuadTree>(Rectangle{ boundary.x, boundary.y, halfWidth, halfHeight });
        children[2] = std::make_unique<QuadTree>(Rectangle{ boundary.x, boundary.y + halfHeight, halfWidth, halfHeight });
        children[3] = std::make_unique<QuadTree>(Rectangle{ boundary.x + halfWidth, boundary.y + halfHeight, halfWidth, halfHeight });

        //std::lock_guard<std::mutex> particlesLock(particlesMutex);

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
        std::lock_guard<std::mutex> particlesLock(particlesMutex);
        std::lock_guard<std::mutex> childrenLock(childrenMutex);
        particles.clear();
        for(int i = 0; i < 4; i++){
            if(children[i]){
                children[i]->clear();
            }
        }
    }

    void Draw() {

        for (auto& p : particles) {
            DrawPixelV(p.pos, p.color);
        }

        //DrawRectangleLinesEx(boundary, 1, GREEN);

        if (isDivided) {
            for (auto& child : children) {
                //std::lock_guard<std::mutex> lock(particlesMutex);
                //std::lock_guard<std::mutex> lock2(childrenMutex);
                if(child->particles.size() > 0){
                    child->Draw();
                }
            }
        }
    }

    void removeParticle(const Particle& p) {
        std::lock_guard<std::mutex> lock(particlesMutex);
        particles.erase(std::remove_if(particles.begin(), particles.end(),
            [&p](const Particle& q) { return &p == &q; }), particles.end());
    }
};

//---------------------------------------------------------------------------------------------------------------------------------

void InsertParticles(const std::vector<Particle>& particles, QuadTree& tree) {
    const int numThreads = 1;
    //std::thread::hardware_concurrency() / 4;
    //std::cout << numThreads << std::endl;
    std::vector<std::vector<Particle>> threadParticles(numThreads);

    // Split the input vector into smaller chunks and assign each chunk to a thread
    const int chunkSize = particles.size() / numThreads;
    for (int i = 0; i < numThreads; i++) {
        const int startIndex = i * chunkSize;
        const int endIndex = (i == numThreads - 1) ? particles.size() : (i + 1) * chunkSize;
        threadParticles[i].reserve(endIndex - startIndex);
        std::copy(particles.begin() + startIndex, particles.begin() + endIndex, std::back_inserter(threadParticles[i]));
    }

    // Process each chunk of particles in a separate thread
    std::vector<std::thread> threads(numThreads);
    for (int i = 0; i < numThreads; i++) {
        threads[i] = std::thread([&tree](const std::vector<Particle>& particles) {
            for (const auto& p : particles) {
                tree.Insert(p);
            }
        }, threadParticles[i]);
    }

    // Wait for all threads to finish and merge their temporary vectors into the main vector
    for (int i = 0; i < numThreads; i++) {
        threads[i].join();
        const std::vector<Particle>& threadParticlesToAdd = threadParticles[i];
        tree.particles.insert(tree.particles.end(), threadParticlesToAdd.begin(), threadParticlesToAdd.end());
    }
}

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
    //Timer t;
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

std::vector<Particle> checkCollisionsMultiThread(std::vector<Particle>& cluster, std::vector<Particle>& FreeParticles, QuadTree& qt) {
    Timer t;
    std::vector<Particle> result;

    qt.clear();
    //qt.InsertAllParticles(FreeParticles);
    //InsertParticles(FreeParticles, qt);
    qt.InsertAllParticlesParallel(FreeParticles);

    // Mutex to protect access to FreeParticles vector
    std::mutex freeParticlesMutex;

    // handle collisions for each particle
    auto handleCollisions = [&](const Particle& p) {
        std::vector<Particle> newParticles = qt.queryCircle(p.pos, collisionThreshold);

        for (auto& np : newParticles) {
            if (CheckCollisionPointCircle(p.pos, np.pos, collisionThreshold / 1.5f)) {
                // Acquire lock before accessing FreeParticles vector
                std::unique_lock<std::mutex> lock(freeParticlesMutex);

                auto it = std::find_if(FreeParticles.begin(), FreeParticles.end(),
                                       [&](const Particle& fp) {
                                           return fp.pos.x == np.pos.x && fp.pos.y == np.pos.y;
                                       });

                if (it != FreeParticles.end()) {
                    // Remove particle from FreeParticles vector
                    FreeParticles.erase(it);
                    lock.unlock();
                }
                result.push_back(np);
            }
        }
    };

    // Create a thread pool with 24 threads
    //const int numThreads = 32;
    std::vector<std::thread> threads(numThreads);

    // Partition the cluster vector into numThreads chunks
    const int chunkSize = (cluster.size() + numThreads - 1) / numThreads;

    // Spawn threads to handle collisions for each chunk of particles
    auto chunkStart = cluster.begin();
    for (int i = 0; i < numThreads; ++i) {
        auto chunkEnd = std::min(chunkStart + chunkSize, cluster.end());
        threads[i] = std::thread([&](auto start, auto end) {
            for (auto it = start; it != end; ++it) {
                handleCollisions(*it);
            }
        }, chunkStart, chunkEnd);
        chunkStart = chunkEnd;
    }

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}

std::vector<Particle> checkCollisions(std::vector<Particle>& cluster, std::vector<Particle>& FreeParticles, QuadTree& qt){
    qt.clear();
    for(unsigned int i = 0; i < FreeParticles.size(); i++){
        qt.Insert(FreeParticles[i]);
    }

    if(false){
        // Split the cluster vector into smaller chunks
        const static int chunkSize = cluster.size() / numThreads;
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

    float queryRadius = 200;

    InitWindow(screenWidth, screenHeight, "DLA");
    SetTargetFPS(100);
    omp_set_num_threads(24);

    //std::vector<Particle> FreeParticles(startingNumParticles,Particle(1000,700,RED));
    std::vector<Particle> FreeParticles = CreateCircle(100,RED,{screenWidth/2.0,screenHeight/2.0}, 30);
    //std::vector<Particle> fp2 = CreateCircle(200000,RED,{screenWidth/2.0,screenHeight/2.0}, queryRadius*2);

    //std::vector<Particle> FreeParticles(1000, Particle(RandomFloat(0,screenWidth, rng),RandomFloat(0,screenHeight, rng), RED));     //random particles
    std::vector<Particle> ClusterParticles(startingClusterParticles,Particle(screenWidth/2.0,screenHeight/2.0,GREEN));

    //std::vector<Particle> ClusterParticles = CreateCircle(5000,WHITE,{screenWidth/2.0,screenHeight/2.0}, queryRadius / 1.2);

    Camera2D camera = { 0 };
    camera.target = { screenWidth / 2.0f, screenHeight / 2.0f };
    camera.offset = { screenWidth / 2.0f, screenHeight / 2.0f };
    camera.rotation = 0.0f;
    camera.zoom = 1.0f;

    QuadTree qt(Rectangle{0,0,screenWidth,screenHeight});

    //main loop
    for (int i = 0; !WindowShouldClose(); i++) {

        if(i % 500 == 0 and i / 5 < screenHeight / 2){
            std::vector<Particle> fp2 = CreateCircle(200 * (1 + i / 50),RED,{screenWidth/2.0,screenHeight/2.0}, 60 + i / 5);
            FreeParticles.insert(FreeParticles.end(), fp2.begin(), fp2.end());
        }

        FreeParticles = randomWalkMultiThreaded(FreeParticles);

        //check for points in central circle using qt
        //std::vector<Particle> centralPoints {qt.queryCircle({screenWidth/2.0,screenHeight/2.0}, queryRadius)};

        //make centralPoints green
        /*for(unsigned int i = 0; i < centralPoints.size(); i++){
            Particle p = centralPoints[i];
            p.color = GREEN;
            centralPoints[i] = p; 
        }*/


        //std::vector<Particle> collided {checkCollisionsMultiThread(ClusterParticles, FreeParticles, qt)};

        qt.clear();
        qt.InsertAllParticlesParallel(FreeParticles);
        //qt.InsertAllParticles(FreeParticles);
        std::vector<Particle> collided = qt.checkCollisions(ClusterParticles);

        for(unsigned int i = 0; i < collided.size(); i++){
            for(unsigned int j = 0; j < FreeParticles.size(); j++){
                if(FreeParticles[j].pos.x == collided[i].pos.x and FreeParticles[j].pos.y == collided[i].pos.y){
                    FreeParticles.erase(FreeParticles.begin() + j);
                }
            }
        }

        for(auto p: collided){
            p.color = WHITE;
            ClusterParticles.push_back(p);
        }


        BeginDrawing();

        ClearBackground(BLACK);
        DrawFPS(20, 20);
        //int s = FreeParticles.size();
        DrawText(TextFormat("FreeParticles: %d\tCluster Particles: %d\tTotal Particles: %d", int(FreeParticles.size()), int(ClusterParticles.size()), int(FreeParticles.size() + ClusterParticles.size())), 200, 100, 20, GREEN);

        //draw cluster
        for (long long unsigned int i = 0; i < ClusterParticles.size(); i++) {
            //DrawRectangleV(ClusterParticles[i].pos, { 2,2 }, ClusterParticles[i].color);
            DrawPixelV(ClusterParticles[i].pos,ClusterParticles[i].color);
        }

        //draw quadtree + freepoints
        qt.Draw();

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
