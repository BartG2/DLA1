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
#include <memory>
#include <list>

std::mt19937 CreateGeneratorWithTimeSeed();
float RandomFloat(float min, float max, std::mt19937& rng);
bool Contains(const Rectangle& r1, const Rectangle& r2);

//-------------------------------------------------------------------------------------------------------------------------------------------------------

constexpr int screenWidth = 2560, screenHeight = 1440, numThreads = 2, maxTreeDepth = 7;
const float collisionThreshold = 3.0f;

const Vector2 particleSize = {2,2};

std::mt19937 rng = CreateGeneratorWithTimeSeed();

//-------------------------------------------------------------------------------------------------------------------------------------------------------

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

class StaticQuadTree {
private:
    const int MAX_CAPACITY = 800; // Maximum number of particles in a node before dividing
    const int MAX_LEVELS = 7; // Maximum depth of the StaticQuadTree

    int level;
    Rectangle bounds;
    
    std::vector<Particle> particles;
    //std::array<std::unique_ptr<StaticQuadTree>, 4> children;
    std::array<std::shared_ptr<StaticQuadTree>, 4> children;
    std::array<Rectangle, 4> childAreas;

public:

    StaticQuadTree(const int currentDepth = 0, const Rectangle& size = {0, 0, screenWidth, screenHeight}){
        level = currentDepth;
        resize(size);
    }

    ~StaticQuadTree() {
        for (int i = 0; i < 4; i++) {
            if (children[i] != nullptr) {
                children[i] = nullptr;
                //delete children[i].release();
            }
        }
    }

    void resize(const Rectangle& newArea){
        clear();

        bounds = newArea;
        float newWidth = bounds.width / 2.0f, newHeight = bounds.height / 2.0f;
        
        //unit circle order
        childAreas[0] = Rectangle{bounds.x + bounds.width, bounds.y, newWidth, newHeight};
        childAreas[1] = Rectangle{bounds.x, bounds.y, newWidth, newHeight};
        childAreas[2] = Rectangle{bounds.x, bounds.y + bounds.height, newWidth, newHeight};
        childAreas[3] = Rectangle{bounds.x + bounds.width, bounds.y + bounds.height, newWidth, newHeight};


    }

    void insert(const Particle& p){
         for(int i = 0; i < 4; i++){
            if(CheckCollisionPointRec(p.pos, childAreas[i])){
                //check if past depth limit
                if(level < MAX_LEVELS){
                    //does child exist
                    if(!children[i]){
                        children[i] = std::make_shared<StaticQuadTree>(level + 1, childAreas[i]);
                    }
                    //recursive calls
                    children[i]->insert(p);
                    return;
                }
            }
         }

         //p can't go in any children, so it goese here
         particles.push_back(p);
    }

    std::list<Particle> search(const Rectangle& searchArea) const{
        std::list<Particle> listItems;
        search(searchArea, listItems);
        return listItems;
    }

    void search(const Rectangle& searchArea, std::list<Particle>& listItems) const{
        //check for Particles belonging to this area and add to list
        for(const auto& p : particles){
            if(CheckCollisionPointRec(p.pos, searchArea)){
                listItems.push_back(p);
            }
        }

        //recursive children calls
        for(int i = 0; i < 4; i++){
            if(children[i]){
                //if child is entirely contained by search area, add all children without bounds check
                if(Contains(searchArea, childAreas[i])){
                    children[i]->returnChildItems(listItems);
                }
                //child and search area overlap, but child not contained
                else if(CheckCollisionRecs(childAreas[i], searchArea)){

                }
            }
        }
    }

    //return all of child's particles
    void returnChildItems(std::list<Particle> listItems){
        for(const auto& p : particles){
            listItems.push_back(p);
        }

        //recursive calls
        for(int i = 0; i < 4; i++){
            if(children[i]){
                children[i]->returnChildItems(listItems);
            }
        }
    }

    void clear(){
        particles.clear();

        for(int i = 0; i < 4; i++){
            if(children[i]){
                children[i]->clear();
            }
            children[i].reset();
        }
    }

    int size(){
        int count = particles.size();

        for(int i = 0; i < 4; i++){
            if(children[i]){
                count += children[i]->size();
            }
        }

        return count;
    }

    void draw(){
        DrawRectangleLinesEx(bounds, 0.5, GREEN);

        for(unsigned int i = 0; i < particles.size(); i++){
            DrawPixelV(particles[i].pos, particles[i].color);
        }

        for(int i = 0; i < 4; i++){
            if(children[i]){
                children[i]->draw();
            }
        }
    }
};

//-------------------------------------------------------------------------------------------------------------------------------------------------------

std::vector<Particle> freeParticles, aggregateParticles;

//-------------------------------------------------------------------------------------------------------------------------------------------------------

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

//r1 contains r2
bool Contains(const Rectangle& r1, const Rectangle& r2){
    return (r2.x >= r1.x) and (r2.x + r2.width < r1.x + r1.width) and (r2.y >= r1.y) and(r2.y + r2.height < r1.y + r1.height);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

void Initialize(){
    InitWindow(screenWidth, screenHeight, "DLA, hopefully");
    SetTargetFPS(100);

    constexpr int startingNumParticles = 100, startingRadius = 100;
    const Color startingColor = RED;
    const Vector2 startingCenter = {screenWidth / 2, screenHeight / 2};

    freeParticles = CreateCircle(startingNumParticles, startingColor, startingCenter, startingRadius);
    aggregateParticles = {1, Particle(screenWidth / 2.0, screenHeight / 2.0, WHITE)};
}

void DrawParticlesVector(const std::vector<Particle>& particles){
    for(unsigned int i = 0; i < particles.size(); i++){
        DrawPixelV(particles[i].pos, particles[i].color);
    }
}

void RandomWalkAll(std::vector<Particle>& particles){
    
    for(auto& p : particles){
        p.RandomWalk(1, 1);
    }
}

StaticQuadTree InitializeQT(){
    StaticQuadTree qt(0, {0, 0, screenWidth, screenHeight});

    for(unsigned int i = 0; i < freeParticles.size(); i++){
        qt.insert(freeParticles[i]);
    }

    return qt;
}

void ConcentricCircles(int frameCount){
    if(frameCount / 5 < screenHeight / 2 and frameCount % 500 == 0){
        std::vector<Particle> fp2 = CreateCircle(40 * (1 + frameCount / 50),RED,{screenWidth/2.0, screenHeight/2.0}, 60 + frameCount / 5);
        freeParticles.insert(freeParticles.end(), fp2.begin(), fp2.end());
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

int main(){
    Initialize();
    
    for(int frameCount = 0; !WindowShouldClose(); frameCount++){

        StaticQuadTree qt = InitializeQT();

        ConcentricCircles(frameCount);

        const Rectangle searchArea {screenWidth / 2, screenHeight / 2, screenWidth / 2, screenHeight / 2};
        std::list<Particle> searchResults = qt.search(searchArea);
        
        BeginDrawing();
        {
            ClearBackground(BLACK);
            DrawFPS(10,10);

            RandomWalkAll(freeParticles);

            DrawParticlesVector(freeParticles);
            DrawParticlesVector(aggregateParticles);
            for(auto it = searchResults.begin(); it != searchResults.end(); it++){
                DrawPixelV(it->pos, BLUE);
            }

        }
        EndDrawing();

    }

    return 0;
}
