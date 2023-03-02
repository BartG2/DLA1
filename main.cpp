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
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>

//---------------------------------------------------------------------------------------------------------------------------------

std::mt19937 CreateGeneratorWithTimeSeed();
float RandomFloat(float min, float max, std::mt19937& rng);

//---------------------------------------------------------------------------------------------------------------------------------


constexpr int screenWidth = 2560, screenHeight = 1440, numThreads = 2, maxTreeDepth = 7;
const float collisionThreshold = 3.0f;

const Vector2 particleSize = {2,2};

std::mt19937 rng = CreateGeneratorWithTimeSeed();

//---------------------------------------------------------------------------------------------------------------------------------

class ThreadPool {
public:
    explicit ThreadPool(int threadPoolNumThreads) : stop(false) {
        for (int i = 0; i < threadPoolNumThreads; ++i) {
            threads.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) {
                            return;
                        }
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (auto& thread : threads) {
            thread.join();
        }
    }

    template<class F, class... Args>
    void enqueue(F&& f, Args&&... args) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        }
        condition.notify_one();
    }

private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

class Particle {
public:
    Vector2 pos;
    Color color;
    bool isStuck;

    Particle(float x, float y, Color col)
        :pos({x,y}), color(col), isStuck(false)
    {}

    Particle()
        :pos({ screenWidth - 10, screenHeight - 10 }), color(WHITE), isStuck(false)
    {}

    void RandomWalk(float stepSize, int numSteps) {
        if(!isStuck){
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
private:
    const int MAX_CAPACITY = 80; // Maximum number of particles in a node before dividing
    const int MAX_LEVELS = 7; // Maximum depth of the QuadTree

    int level;
    Rectangle bounds;
    std::vector<Particle> particles;
    std::array<std::unique_ptr<QuadTree>, 4> children;
    //QuadTree* children[4];

    bool IsDivisible() {
        return level < MAX_LEVELS and particles.size() > MAX_CAPACITY;
    }

    void Subdivide() {
        float x = bounds.x;
        float y = bounds.y;
        float w = bounds.width / 2.0f;
        float h = bounds.height / 2.0f;

        children[0] = std::make_unique<QuadTree>(level + 1, Rectangle{x, y, w, h});
        children[1] = std::make_unique<QuadTree>(level + 1, Rectangle{x + w, y, w, h});
        children[2] = std::make_unique<QuadTree>(level + 1, Rectangle{x + w, y + h, w, h});
        children[3] = std::make_unique<QuadTree>(level + 1, Rectangle{x, y + h, w, h});

        for(unsigned int i = 0; i < particles.size(); i++){
            for(unsigned int j = 0; j < 4; j++){
                if(CheckCollisionPointRec(particles[i].pos, bounds)){
                    children[j]->Insert(particles[i]);
                    break;
                }
            }
        }

        particles.clear();
    }

public:
    /*QuadTree(int level, Rectangle bounds)
        :level(level), bounds(bounds)
    {
        for (int i = 0; i < 4; i++) {
            children[i] = nullptr;
        }
    }*/

    QuadTree(int level, Rectangle bounds)
        : level(level), bounds(bounds), children()
    {}

    ~QuadTree() {
        for (int i = 0; i < 4; i++) {
            if (children[i] != nullptr) {
                delete children[i].release();
            }
        }
    }

    void Insert(const Particle& particle) {
        if(!CheckCollisionPointRec(particle.pos, bounds)){
            return;
        }

        if (IsDivisible()) {
            if (children[0] == nullptr) {
                Subdivide();
            }

            for (int i = 0; i < 4; i++) {
                children[i]->Insert(particle);
            }
        }
        else {
            particles.push_back(particle);
        }
    }

    std::vector<Particle> Query(const Vector2 center, const float radius) {
        std::vector<Particle> result;

        // Check if the node's bounds intersect with the circle
        if (!CheckCollisionCircleRec(center, radius, bounds)) {
            return result;
        }

        // If the node is not divisible, check each particle and add it to the result if it is inside the circle
        if (!IsDivisible()) {
            for (const auto& particle : particles) {
                if (CheckCollisionPointCircle(particle.pos, center, radius)) {
                    result.push_back(particle);
                }
            }
        }
        else {
            // If the node is divisible, recursively query each child node and concatenate their results
            for (const auto& child : children) {
                if (child != nullptr) {
                    std::vector<Particle> childResult = child->Query(center, radius);
                    result.insert(result.end(), childResult.begin(), childResult.end());
                }
            }
        }


        return result;
    }

    void clear(){
        particles.clear();

        if(children[0] != nullptr){
            for(const auto& child : children){
                child->clear();
            }
        }
    }

    /*std::vector<Particle> Query(const Rectangle& range) const {
        std::vector<Particle> result;

        if (!bounds.Intersects(range)) {
            return result;
        }

        for (const auto& particle : particles) {
            if (range.Contains(particle.pos)) {
                result.push_back(particle);
            }
        }

        if (children[0] != nullptr) {
            for (int i = 0; i < 4; i++) {
                auto childResult = children[i]->Query(range);
                result.insert(result.end(), childResult.begin(), childResult.end());
            }
        }

        return result;
    }*/

    void Draw(){
        /*DrawRectangleLinesEx(bounds, 0.5, GREEN);
        if(!IsDivisible()){
            DrawText(TextFormat("%d", int(particles.size())), bounds.x + 2 * level, bounds.y + 2 * level, 1, GREEN);
        }*/

        for(unsigned int i = 0; i < particles.size(); i++){
            DrawPixelV(particles[i].pos, particles[i].color);
        }

        if(IsDivisible()){
            for(int i = 0; i < 4; i++){
                if(children[i] != nullptr){
                    children[i]->Draw();
                }
            }
        }
    }

    int getTreeSize(){
        int count = particles.size();
        if(children[0] != nullptr){
            for(const auto& child : children){
                count += child->particles.size();
            }
        }
        std::cout << count << std::endl;
        return count;
    }
};

//---------------------------------------------------------------------------------------------------------------------------------

std::vector<Particle> freeParticles, aggregateParticles;

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
//---------------------------------------------------------------------------------------------------------------------------------

int main() {

    InitWindow(screenWidth, screenHeight, "DLA");
    SetTargetFPS(100);
    omp_set_num_threads(32);

    freeParticles = CreateCircle(400,RED,{screenWidth/2.0,screenHeight/2.0}, 40);
    aggregateParticles = {1, Particle(screenWidth / 2.0, screenHeight / 2.0, GREEN)};

    //main loop
    for (int frameCount = 0; !WindowShouldClose(); frameCount++) {

        //make concentric circles of particles
        if(frameCount / 5 < screenHeight / 2 and frameCount % 500 == 0){
            std::vector<Particle> fp2 = CreateCircle(100 * (1 + frameCount / 50),RED,{screenWidth/2.0, screenHeight/2.0}, 60 + frameCount / 5);
            freeParticles.insert(freeParticles.end(), fp2.begin(), fp2.end());
        }
        //random walk for each
        for(unsigned int i = 0; i < freeParticles.size(); i++){
            Particle p = freeParticles[i];
            p.RandomWalk(1, 1);
            freeParticles[i] = p;
        }

        //make quad tree
        QuadTree qt {0, Rectangle{0, 0, screenWidth, screenHeight}}; 
        for(unsigned int i = 0; i < freeParticles.size(); i++){
            qt.Insert(freeParticles[i]);
        }

        BeginDrawing();
        {

            ClearBackground(BLACK);
            DrawFPS(20,20);
            DrawText(TextFormat("freeParticles: %d,\tCluster Particles: %d,\tTotal Particles: %d", int(freeParticles.size()), int(aggregateParticles.size()), int(freeParticles.size() + aggregateParticles.size())), 20, 50, 20, GREEN);

            qt.Draw();
            
            //draw cluster
            for (unsigned int i = 0; i < aggregateParticles.size(); i++) {
                DrawPixelV(aggregateParticles[i].pos, aggregateParticles[i].color);
            }
        }
        EndDrawing();
    }

    CloseWindow();

    return 0;
}

