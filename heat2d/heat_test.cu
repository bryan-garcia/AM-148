#include "heat2D.h"
#include <vector>

int main() {

    vector<int> grid_resolution {16, 32, 64, 128, 256, 512, 1024};

    float2 xlim = {-0.5f, 0.5f};
    float2 ylim = {-0.5f, 0.5f};

    string fname = "sol";
    int version = 1;

    ofstream fbenchmark;
    fbenchmark.open("benchmark_times.dat");

    ofstream fout;
    fout.open("iter_val.dat");

    for (int M : grid_resolution) {

        float tcurr = 0.0;
        float tmax = 2.0;
        float tio = 0.5;

        Heat2D Grid(xlim, ylim, M, tmax);
        Grid.initialize_grid();

        if (M == 128) {
            Grid.update_fval();
            Grid.save_fval("sol0.dat");
        }

        while (tcurr < tmax) { 

            if (fmod(tcurr, tio) == 0.0f and (M == 128)) {
                Grid.update_fval();
                Grid.save_fval(fname + to_string(version) + ".dat");
                version++;
            }
            
            Grid.FTCS();
            Grid.BC();
            tcurr += Grid.dt;
        }

        if(fmod(tcurr, tio) != 0.0f and (M == 128)) {
            Grid.update_fval();	
            Grid.save_fval(fname + to_string(version) + ".dat");
        }

        if (M == 128) {
            Grid.save_x("x_val.dat");
            Grid.save_y("y_val.dat");
            fout << version << endl;
            fout.close();
        }

        Grid.deallocate();

        fbenchmark << Grid.execution_time;
        if (M != 1024) fbenchmark << ", ";

        cout << "-----------------------------------------------------------------------------------\n";
        cout << "GRID SIZE: " << M << " x " << M << endl;
        cout << "Cumulative FTCS and BC execution time: " << Grid.execution_time << " (ms)" << endl;
        cout << "-----------------------------------------------------------------------------------\n";

	}



    fbenchmark.close();

    return 0;
}
