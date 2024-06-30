#include <iostream>
#include <algorithm>
#include <tuple>
#include <vector>
#include <queue>
#include <utility>
#include <random>
#include <chrono>

enum class MoveType {
    VALID_EMPTY,     // Valid move to empty cell
    VALID_OBSTACLE,  // Valid move to obstacle cell, thus disabling it
    INVALID          // Invalid move
};

class ChessBoard {
public:
    ChessBoard(int size = 8) 
        : size(size), board(size, std::vector<char>(size, ' ')), start(-1, -1), end(-1, -1) {}

    void setStart(int x, int y) {
        start = {x, y};
        board[x][y] = 'S';
    }

    void setEnd(int x, int y) {
        end = {x, y};
        board[x][y] = 'E';
    }

    void addObstacle(int x, int y, char obstacle = 'X') {
        obstacles.push_back({x, y});
        board[x][y] = obstacle;
    }

    MoveType is_valid_move(int x, int y, int prev_x, int prev_y) const {
        bool is_obstacle = false;
        // border check
        if (x < 0 || x >= size || y < 0 || y >= size) return MoveType::INVALID;
        // has own side obstacle (asserting we are Black)
        if (board[x][y] == 'B') return MoveType::INVALID;
        else if (board[x][y] == 'R') is_obstacle = true;

        // check blocking obstacles
        int block_x = prev_x, block_y = prev_y, dx = x - prev_x, dy = y - prev_y;
        if (abs(dx) == 2) {
            block_x = (x + prev_x) / 2;
            if (dy == 1 || dy == -1) 
                if (board[block_x][y] == ' ' && board[block_x][prev_y] == ' ') 
                    return is_obstacle ? MoveType::VALID_OBSTACLE : MoveType::VALID_EMPTY;
        }
        else if (abs(dy) == 2) {
            block_y = (y + prev_y) / 2;
            if (dx == 1 || dx == -1) 
                if (board[x][block_y] == ' ' && board[prev_x][block_y] == ' ') 
                    return is_obstacle ? MoveType::VALID_OBSTACLE : MoveType::VALID_EMPTY;
        }
        
        return MoveType::INVALID;
    }

    bool isGoalReached(int x, int y) {
        return std::make_pair(x, y) == end;
    }

    std::vector<std::tuple<int, int, bool>> getPossibleMoves(int x, int y) const {
        std::vector<std::pair<int, int>> possible_moves = {
            {x + 1, y + 2}, {x + 2, y + 1},
            {x + 2, y - 1}, {x + 1, y - 2},
            {x - 1, y - 2}, {x - 2, y - 1},
            {x - 2, y + 1}, {x - 1, y + 2}
        };
        // valid_moves: (x, y, is_obstacle)
        std::vector<std::tuple<int, int, bool>> valid_moves;
        for (auto& move : possible_moves) {
            auto movetype = is_valid_move(move.first, move.second, x, y);
            if (movetype == MoveType::VALID_EMPTY) {
                valid_moves.push_back({move.first, move.second, false});
            }
            else if (movetype == MoveType::VALID_OBSTACLE) {
                valid_moves.push_back({move.first, move.second, true});
            }
        }
        return valid_moves;
    }

    void printBoard() const {
        std::cout << "+";
        for (int i = 0; i < size; ++i) std::cout << "---+";
        std::cout << std::endl;
        for (auto& row : board) {
            std::cout << "|";
            for (char cell : row) {
                std::cout << " " << cell << " |";
            }
            std::cout << "\n+";
            for (int i = 0; i < size; ++i) std::cout << "---+";
            std::cout << std::endl;
        }
    }

    void generateRandomObstacles(float obstaclesRatio = 0.3, unsigned seed = 0) {
        int num_obstacles = static_cast<int>(obstaclesRatio * size * size);
        std::mt19937 rng(seed); // Standard mersenne_twister_engine seeded with seed
        std::uniform_int_distribution<> dist(0, size - 1);
        std::uniform_real_distribution<> prob(0, 1);

        for (int i = 0; i < num_obstacles; ++i) {
            int x = dist(rng);
            int y = dist(rng);
            if (std::make_pair(x, y) != start && std::make_pair(x, y) != end && 
                std::find(obstacles.begin(), obstacles.end(), std::make_pair(x, y)) == obstacles.end()) {
                if (prob(rng) < 0.5) addObstacle(x, y, 'R');  // Red obstacle
                else addObstacle(x, y, 'B');                  // Black obstacle
            }
        }
    }

    void solveBFS(bool detailed_output = true) {  // breadth-first search
        auto global_backup_board = board;  // for restoring the board after the search
        // open list, with each element as (x, y, disabled_obstacles)
        std::queue<std::tuple<int, int, std::vector<std::pair<int, int>>>> q;
        // for backtracking the path
        std::vector<std::vector<std::pair<int, int>>> prev(size, std::vector<std::pair<int, int>>(size, {-1, -1}));
        
        q.push({start.first, start.second, {}});   // Start node, no disabled obstacles
        prev[start.first][start.second] = {0, 0};  // Self-pointing to indicate start

        while (!q.empty()) {
            // get a node from the open list
            auto [x, y, disabled_obstacles] = q.front();
            q.pop();

            if (isGoalReached(x, y)) {
                // End found
                // reconstruct path
                std::vector<std::pair<int, int>> path;
                while (!(x == start.first && y == start.second)) {
                    path.push_back({x, y});
                    auto [px, py] = prev[x][y];
                    x = px;
                    y = py;
                }
                path.push_back(start); // Add start to the path
                std::reverse(path.begin(), path.end()); // Reverse the path to start->end direction

                // Print path
                std::cout << "Path length: " << path.size() - 1 << std::endl;
                if (detailed_output) {
                    std::cout << "Path from Start to End:" << std::endl;
                    for (auto [px, py] : path) {
                        std::cout << "(" << px << ", " << py << ") ";
                    }
                    std::cout << std::endl;
                }

                // Print board with path
                if (detailed_output) {
                    std::cout << "Board with Path:" << std::endl;
                    for (auto [ox, oy] : disabled_obstacles) {
                        board[ox][oy] = ' ';
                    }
                    for (auto [px, py] : path) {
                        board[px][py] = '.';
                    }
                    printBoard();
                }
                board = global_backup_board;  // restore the board
                return;
            }

            // Get all possible moves from current position
            // disable all obstacles that are recorded in disabled_obstacles
            auto backup_board = board;
            for (auto [ox, oy] : disabled_obstacles) {
                board[ox][oy] = ' ';
            }
            auto moves = getPossibleMoves(x, y);
            board = backup_board;
            for (auto [nx, ny, is_obstacle] : moves) {
                auto new_disabled_obstacles = disabled_obstacles;
                if (is_obstacle) new_disabled_obstacles.push_back({nx, ny});
                if (prev[nx][ny] == std::make_pair(-1, -1)) { // Check if not visited
                    q.push({nx, ny, new_disabled_obstacles}); // Add to open list
                    prev[nx][ny] = {x, y}; // Set current as previous for next position
                }
            }
        }

        std::cout << "No path found from Start to End." << std::endl;
        board = global_backup_board;  // restore the board
    }

    bool cmp_greater(const std::tuple<int, int, std::vector<std::pair<int, int>>>& a, const std::tuple<int, int, std::vector<std::pair<int, int>>>& b) const {
        int dx1 = std::get<0>(a) - end.first, dy1 = std::get<1>(a) - end.second;
        int dx2 = std::get<0>(b) - end.first, dy2 = std::get<1>(b) - end.second;
        return dx1 * dx1 + dy1 * dy1 > dx2 * dx2 + dy2 * dy2;
    }

    void solveGBFS(bool detailed_output = true) {  // greedy-best-first search
        auto global_backup_board = board;  // for restoring the board after the search
        auto cmp = [this](const auto& a, const auto& b) { return this->cmp_greater(a, b); };
        // open list, with each element as (x, y, disabled_obstacles)
        std::priority_queue<std::tuple<int, int, std::vector<std::pair<int, int>>>,
                            std::vector<std::tuple<int, int, std::vector<std::pair<int, int>>>>,
                            decltype(cmp)> q(cmp);
        // for backtracking the path
        std::vector<std::vector<std::pair<int, int>>> prev(size, std::vector<std::pair<int, int>>(size, {-1, -1}));
        
        q.push({start.first, start.second, {}});   // Start node, no disabled obstacles
        prev[start.first][start.second] = {0, 0};  // Self-pointing to indicate start

        while (!q.empty()) {
            // get a node from the open list
            auto [x, y, disabled_obstacles] = q.top();
            q.pop();

            if (isGoalReached(x, y)) {
                // End found
                // reconstruct path
                std::vector<std::pair<int, int>> path;
                while (!(x == start.first && y == start.second)) {
                    path.push_back({x, y});
                    auto [px, py] = prev[x][y];
                    x = px;
                    y = py;
                }
                path.push_back(start); // Add start to the path
                std::reverse(path.begin(), path.end()); // Reverse the path to start->end direction

                // Print path
                std::cout << "Path length: " << path.size() - 1 << std::endl;
                if (detailed_output) {
                    std::cout << "Path from Start to End:" << std::endl;
                    for (auto [px, py] : path) {
                        std::cout << "(" << px << ", " << py << ") ";
                    }
                    std::cout << std::endl;
                }

                // Print board with path
                if (detailed_output) {
                    std::cout << "Board with Path:" << std::endl;
                    for (auto [ox, oy] : disabled_obstacles) {
                        board[ox][oy] = ' ';
                    }
                    for (auto [px, py] : path) {
                        board[px][py] = '.';
                    }
                    printBoard();
                }
                board = global_backup_board;  // restore the board
                return;
            }

            // Get all possible moves from current position
            // disable all obstacles that are recorded in disabled_obstacles
            auto backup_board = board;
            for (auto [ox, oy] : disabled_obstacles) {
                board[ox][oy] = ' ';
            }
            auto moves = getPossibleMoves(x, y);
            board = backup_board;
            for (auto [nx, ny, is_obstacle] : moves) {
                auto new_disabled_obstacles = disabled_obstacles;
                if (is_obstacle) new_disabled_obstacles.push_back({nx, ny});
                if (prev[nx][ny] == std::make_pair(-1, -1)) { // Check if not visited
                    q.push({nx, ny, new_disabled_obstacles}); // Add to open list
                    prev[nx][ny] = {x, y}; // Set current as previous for next position
                }
            }
        }

        std::cout << "No path found from Start to End." << std::endl;
        board = global_backup_board;  // restore the board
    }


    bool eval_greater(const std::tuple<int, int, std::vector<std::pair<int, int>>, int>& a, const std::tuple<int, int, std::vector<std::pair<int, int>>, int>& b) const {
        int dx1 = std::get<0>(a) - end.first, dy1 = std::get<1>(a) - end.second;
        int dx2 = std::get<0>(b) - end.first, dy2 = std::get<1>(b) - end.second;
        return dx1 + dy1 + std::get<3>(a) > dx2 + dy2 + std::get<3>(b);
    }



    void solveAStar(bool detailed_output = true) {  // A* search
        auto global_backup_board = board;  // for restoring the board after the search
        auto cmp = [this](const auto& a, const auto& b) { return this->eval_greater(a, b); };
        // open list, with each element as (x, y, disabled_obstacles, steps)
        std::priority_queue<std::tuple<int, int, std::vector<std::pair<int, int>>, int>,
                            std::vector<std::tuple<int, int, std::vector<std::pair<int, int>>, int>>,
                            decltype(cmp)> q(cmp);
        // for backtracking the path
        std::vector<std::vector<std::pair<int, int>>> prev(size, std::vector<std::pair<int, int>>(size, {-1, -1}));
        
        q.push({start.first, start.second, {}, 0});   // Start node, no disabled obstacles
        prev[start.first][start.second] = {0, 0};  // Self-pointing to indicate start

        while (!q.empty()) {
            // get a node from the open list
            auto [x, y, disabled_obstacles, steps] = q.top();
            q.pop();

            if (isGoalReached(x, y)) {
                // End found
                // reconstruct path
                std::vector<std::pair<int, int>> path;
                while (!(x == start.first && y == start.second)) {
                    path.push_back({x, y});
                    auto [px, py] = prev[x][y];
                    x = px;
                    y = py;
                }
                path.push_back(start); // Add start to the path
                std::reverse(path.begin(), path.end()); // Reverse the path to start->end direction

                // Print path
                std::cout << "Path length: " << path.size() - 1 << std::endl;
                if (detailed_output) {
                    std::cout << "Path from Start to End:" << std::endl;
                    for (auto [px, py] : path) {
                        std::cout << "(" << px << ", " << py << ") ";
                    }
                    std::cout << std::endl;
                }

                // Print board with path
                if (detailed_output) {
                    std::cout << "Board with Path:" << std::endl;
                    for (auto [ox, oy] : disabled_obstacles) {
                        board[ox][oy] = ' ';
                    }
                    for (auto [px, py] : path) {
                        board[px][py] = '.';
                    }
                    printBoard();
                }
                board = global_backup_board;  // restore the board
                return;
            }

            // Get all possible moves from current position
            // disable all obstacles that are recorded in disabled_obstacles
            auto backup_board = board;
            for (auto [ox, oy] : disabled_obstacles) {
                board[ox][oy] = ' ';
            }
            auto moves = getPossibleMoves(x, y);
            board = backup_board;
            for (auto [nx, ny, is_obstacle] : moves) {
                auto new_disabled_obstacles = disabled_obstacles;
                if (is_obstacle) new_disabled_obstacles.push_back({nx, ny});
                if (prev[nx][ny] == std::make_pair(-1, -1)) { // Check if not visited
                    q.push({nx, ny, new_disabled_obstacles, steps + 1}); // Add to open list
                    prev[nx][ny] = {x, y}; // Set current as previous for next position
                }
            }
        }

        std::cout << "No path found from Start to End." << std::endl;
        board = global_backup_board;  // restore the board
    }

private:
    int size;
    std::vector<std::vector<char>> board;
    std::pair<int, int> start, end;
    std::vector<std::pair<int, int>> obstacles;
};

int main() {
    int size = 8;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    ChessBoard chessboard(size);
    chessboard.setStart(0, 0);
    chessboard.setEnd(size - 1, size - 1);
    chessboard.generateRandomObstacles(0.3, seed);

    std::cout << "Chess Board Size: " << size << "x" << size << std::endl;
    std::cout << "Initial Board:" << std::endl;
    chessboard.printBoard();

    std::cout << "Solving using Breadth-First Search:" << std::endl;
    std::chrono::steady_clock::time_point search_begin = std::chrono::steady_clock::now();
    chessboard.solveBFS();
    std::chrono::steady_clock::time_point search_end = std::chrono::steady_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_begin);
    std::cout << "Search Time: " << search_time.count() << " us" << std::endl;

    std::cout << "Solving using Greedy-Best-First Search:" << std::endl;
    search_begin = std::chrono::steady_clock::now();
    chessboard.solveGBFS();
    search_end = std::chrono::steady_clock::now();
    search_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_begin);
    std::cout << "Search Time: " << search_time.count() << " us" << std::endl;

    std::cout << "Solving using A* Search:" << std::endl;
    search_begin = std::chrono::steady_clock::now();
    chessboard.solveAStar();
    search_end = std::chrono::steady_clock::now();
    search_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_begin);
    std::cout << "Search Time: " << search_time.count() << " us" << std::endl;



    size = 100;
    seed = std::chrono::system_clock::now().time_since_epoch().count();
    ChessBoard chessboard_large(size);
    chessboard_large.setStart(0, 0);
    chessboard_large.setEnd(size - 1, size - 1);
    chessboard_large.generateRandomObstacles(0.3, seed);

    std::cout << "Chess Board Size: " << size << "x" << size << std::endl;

    std::cout << "Solving using Breadth-First Search:" << std::endl;
    search_begin = std::chrono::steady_clock::now();
    chessboard_large.solveBFS(false);
    search_end = std::chrono::steady_clock::now();
    search_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_begin);
    std::cout << "Search Time: " << search_time.count() << " us" << std::endl;

    std::cout << "Solving using Greedy-Best-First Search:" << std::endl;
    search_begin = std::chrono::steady_clock::now();
    chessboard_large.solveGBFS(false);
    search_end = std::chrono::steady_clock::now();
    search_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_begin);
    std::cout << "Search Time: " << search_time.count() << " us" << std::endl;

    std::cout << "Solving using A* Search:" << std::endl;
    search_begin = std::chrono::steady_clock::now();
    chessboard_large.solveAStar(false);
    search_end = std::chrono::steady_clock::now();
    search_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_begin);
    std::cout << "Search Time: " << search_time.count() << " us" << std::endl;



    size = 1000;
    seed = std::chrono::system_clock::now().time_since_epoch().count();
    ChessBoard chessboard_largest(size);
    chessboard_largest.setStart(0, 0);
    chessboard_largest.setEnd(size - 1, size - 1);
    chessboard_largest.generateRandomObstacles(0.3, seed);

    std::cout << "Chess Board Size: " << size << "x" << size << std::endl;

    // // This takes very long (~3min) to solve

    // std::cout << "Solving using Breadth-First Search:" << std::endl;
    // search_begin = std::chrono::steady_clock::now();
    // chessboard_largest.solveBFS(false);
    // search_end = std::chrono::steady_clock::now();
    // search_time = std::chrono::duration_cast<std::chrono::seconds>(search_end - search_begin);
    // std::cout << "Search Time: " << search_time.count() << " s" << std::endl;

    std::cout << "Solving using Greedy-Best-First Search:" << std::endl;
    search_begin = std::chrono::steady_clock::now();
    chessboard_largest.solveGBFS(false);
    search_end = std::chrono::steady_clock::now();
    search_time = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_begin);
    std::cout << "Search Time: " << search_time.count() << " ms" << std::endl;

    std::cout << "Solving using A* Search:" << std::endl;
    search_begin = std::chrono::steady_clock::now();
    chessboard_largest.solveAStar(false);
    search_end = std::chrono::steady_clock::now();
    search_time = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_begin);
    std::cout << "Search Time: " << search_time.count() << " ms" << std::endl;
    
    return 0;
}
