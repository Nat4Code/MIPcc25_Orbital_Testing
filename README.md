# MIPcc25_Orbital_Testing
Testing Various Orbital Branching Heuristics and Exact Methods for Nonlinear Programs

Compilation for orbit helper:
g++ -O3 -DNDEBUG -march=native -flto -std=c++20   -I"$PWD/local/bliss/include"   orbit_helper.cpp -o orbit_helper   -L"$PWD/local/bliss/lib" -lbliss   -Wl,-rpath,"$PWD/local/bliss/lib"

Compilation and run command for orbit exact:
g++ -O3 -DNDEBUG -march=native -flto -std=c++20 \
    -I/opt/gurobi1203/linux64/include \
    orbit_branch_exact_bliss.cpp -o orbit_branch_exact \
    -L/opt/gurobi1203/linux64/lib -lgurobi_c++ -lgurobi120

    g++ -O3 -DNDEBUG -march=native -flto -std=c++20 \
    -I/opt/gurobi1303/linux64/include \
    orbit_branch_exact_bliss.cpp -o orbit_branch_exact \
    -L/opt/gurobi1301/linux64/lib -lgurobi_c++ -lgurobi130 \
    -Wl,-rpath,/opt/gurobi1301/linux64/lib

./orbit_branch_exact Initial_problem_set/Test_prob_4.mps

Compilation and run command for orbit depth:
    g++ -O3 -DNDEBUG -march=native -flto -std=c++20 \
    -I/opt/gurobi1301/linux64/include \
    orbit_branch_depth_bliss.cpp -o orbit_branch_depth \
    -L/opt/gurobi1301/linux64/lib -lgurobi_c++ -lgurobi130 \
    -Wl,-rpath,/opt/gurobi1301/linux64/lib

./orbit_branch_depth Initial_problem_set/Test_prob_4.mps

Compilation and run command for orbit sensing:
    g++ -O3 -DNDEBUG -march=native -flto -std=c++20 \
    -I/opt/gurobi1301/linux64/include \
    orbit_branch_sensing_bliss.cpp -o orbit_branch_sensing \
    -L/opt/gurobi1301/linux64/lib -lgurobi_c++ -lgurobi130 \
    -Wl,-rpath,/opt/gurobi1301/linux64/lib