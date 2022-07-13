#include <stdio.h>
#include <stdlib.h>
#include <rebound.h>

#include "../forces.h"
#include "../profiles.h"
#include "../common.h"
#include "../drag_force.h"

double cAU = 14959787070000;
double cYR = 365.2564 * 86400;
double cMS = 1.9885e33;

// double cAU = 1.0;
// double cYR = 1.0;
// double cMS = 1.0;

#define cV cAU / cYR
#define cE cMS * cAU * cAU / (cYR * cYR)
#define cD cMS / (cAU * cAU * cAU)

double aI;
double eI;
double EI;
double TI = 0.0;
double diag_time = -0.0;

void heartbeat(struct reb_simulation *reb_sim) {
    if (reb_sim->t > diag_time) {
        struct IOPF_drag_force_diag diag;
        struct reb_particle *p = reb_sim->particles + 1;
        IOPF_drag_force(p, reb_sim->particles, &diag);

        diag_time=reb_sim->t+(reb_sim->t < 2 ? 0.05 : 250.0);

        double E = reb_tools_energy(reb_sim);
        fprintf(stdout, "t=%f, ΔE/Δt=%.5e\n", reb_sim->t, (E-EI)/(reb_sim->t-TI) * cE);
        fprintf(stdout, "Planet: d=%.5f, x=%.5e, y=%.5e, vx=%.5e, vy=%.5e, v=%.5e.\n", diag.orbit.d, p->x * cAU, p->y * cAU, p->vx * cV, p->vy * cV, diag.orbit.v * cV);
        fprintf(stdout, "Interpolated: vt_gas=%.5e, vr_gas=%.5e, rho_0=%.5e\n", diag.vt_gas * cV, diag.vr_gas * cV, diag.rho_0 * cD);
        fprintf(stdout, "Gas: vx=%.5e, vy=%.5e, v=%.5e. Relative: v=%.5e, ux=%.5f, uy=%.5f\n", diag.vx_gas * cV, diag.vy_gas * cV, diag.v_gas * cV, diag.v_rel * cV, diag.ux_rel, diag.uy_rel);
        fprintf(stdout, "Drag acceleration: a=%.5e, dE/dt=%.5e\n", diag.a_d * cV / cYR, diag.P_d * cE / cYR);
        fprintf(stdout, "Orbit: a=%.10e, e=%.10e, Δa/Δt=%.10e, Δe/Δt=%.10e\n\n", diag.orbit.a, diag.orbit.e, (diag.orbit.a-aI)/(reb_sim->t-TI), (diag.orbit.e-eI)/(reb_sim->t-TI));
        fflush(stdout);

        aI = diag.orbit.a;
        eI = diag.orbit.e;
        EI=E;
        TI=reb_sim->t;

        reb_move_to_com(reb_sim);
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <m> <a> <e> <C_D>\n", argv[0]);
        exit(1);
    }

    double* xp = STD_PROF_X; double* vtp = VELOCITY_PROF[0]; double* vrp = VELOCITY_PROF[1]; double* dp = DENSITY_PROF;
    unsigned int n = 0;
    *xp++ = 2.000000e-02; *vtp++ = 2.107132e+07 * CM_PER_S; *vrp++ = 2.246802e+03 * CM_PER_S; *dp++ = 1.620334e-09 * G_PER_CM3; n++;
    *xp++ = 2.058000e-02; *vtp++ = 2.077247e+07 * CM_PER_S; *vrp++ = 2.114639e+03 * CM_PER_S; *dp++ = 1.599020e-09 * G_PER_CM3; n++;
    *xp++ = 2.117600e-02; *vtp++ = 2.047768e+07 * CM_PER_S; *vrp++ = 1.994472e+03 * CM_PER_S; *dp++ = 1.576198e-09 * G_PER_CM3; n++;
    *xp++ = 2.179000e-02; *vtp++ = 2.018698e+07 * CM_PER_S; *vrp++ = 1.884303e+03 * CM_PER_S; *dp++ = 1.552227e-09 * G_PER_CM3; n++;
    *xp++ = 2.242100e-02; *vtp++ = 1.990056e+07 * CM_PER_S; *vrp++ = 1.782587e+03 * CM_PER_S; *dp++ = 1.527690e-09 * G_PER_CM3; n++;
    *xp++ = 2.307100e-02; *vtp++ = 1.961830e+07 * CM_PER_S; *vrp++ = 1.688110e+03 * CM_PER_S; *dp++ = 1.502769e-09 * G_PER_CM3; n++;
    *xp++ = 2.373900e-02; *vtp++ = 1.933996e+07 * CM_PER_S; *vrp++ = 1.598422e+03 * CM_PER_S; *dp++ = 1.479943e-09 * G_PER_CM3; n++;
    *xp++ = 2.442700e-02; *vtp++ = 1.906561e+07 * CM_PER_S; *vrp++ = 1.513216e+03 * CM_PER_S; *dp++ = 1.458850e-09 * G_PER_CM3; n++;
    *xp++ = 2.513500e-02; *vtp++ = 1.879516e+07 * CM_PER_S; *vrp++ = 1.433063e+03 * CM_PER_S; *dp++ = 1.438096e-09 * G_PER_CM3; n++;
    *xp++ = 2.586400e-02; *vtp++ = 1.852855e+07 * CM_PER_S; *vrp++ = 1.357430e+03 * CM_PER_S; *dp++ = 1.417788e-09 * G_PER_CM3; n++;
    *xp++ = 2.661300e-02; *vtp++ = 1.826573e+07 * CM_PER_S; *vrp++ = 1.285878e+03 * CM_PER_S; *dp++ = 1.398172e-09 * G_PER_CM3; n++;
    *xp++ = 2.738400e-02; *vtp++ = 1.800666e+07 * CM_PER_S; *vrp++ = 1.218037e+03 * CM_PER_S; *dp++ = 1.379227e-09 * G_PER_CM3; n++;
    *xp++ = 2.817800e-02; *vtp++ = 1.775126e+07 * CM_PER_S; *vrp++ = 1.153601e+03 * CM_PER_S; *dp++ = 1.361002e-09 * G_PER_CM3; n++;
    *xp++ = 2.899500e-02; *vtp++ = 1.749949e+07 * CM_PER_S; *vrp++ = 1.092306e+03 * CM_PER_S; *dp++ = 1.343610e-09 * G_PER_CM3; n++;
    *xp++ = 2.983500e-02; *vtp++ = 1.725130e+07 * CM_PER_S; *vrp++ = 1.033930e+03 * CM_PER_S; *dp++ = 1.327147e-09 * G_PER_CM3; n++;
    *xp++ = 3.069900e-02; *vtp++ = 1.700664e+07 * CM_PER_S; *vrp++ = 9.782615e+02 * CM_PER_S; *dp++ = 1.311670e-09 * G_PER_CM3; n++;
    *xp++ = 3.158900e-02; *vtp++ = 1.676545e+07 * CM_PER_S; *vrp++ = 9.251292e+02 * CM_PER_S; *dp++ = 1.297145e-09 * G_PER_CM3; n++;
    *xp++ = 3.250400e-02; *vtp++ = 1.652766e+07 * CM_PER_S; *vrp++ = 8.744195e+02 * CM_PER_S; *dp++ = 1.283652e-09 * G_PER_CM3; n++;
    *xp++ = 3.344600e-02; *vtp++ = 1.629327e+07 * CM_PER_S; *vrp++ = 8.260061e+02 * CM_PER_S; *dp++ = 1.271136e-09 * G_PER_CM3; n++;
    *xp++ = 3.441600e-02; *vtp++ = 1.606225e+07 * CM_PER_S; *vrp++ = 7.797775e+02 * CM_PER_S; *dp++ = 1.259606e-09 * G_PER_CM3; n++;
    *xp++ = 3.541300e-02; *vtp++ = 1.583447e+07 * CM_PER_S; *vrp++ = 7.356340e+02 * CM_PER_S; *dp++ = 1.249178e-09 * G_PER_CM3; n++;
    *xp++ = 3.643900e-02; *vtp++ = 1.560993e+07 * CM_PER_S; *vrp++ = 6.934862e+02 * CM_PER_S; *dp++ = 1.239800e-09 * G_PER_CM3; n++;
    *xp++ = 3.749500e-02; *vtp++ = 1.538858e+07 * CM_PER_S; *vrp++ = 6.532530e+02 * CM_PER_S; *dp++ = 1.231479e-09 * G_PER_CM3; n++;
    *xp++ = 3.858200e-02; *vtp++ = 1.517037e+07 * CM_PER_S; *vrp++ = 6.148599e+02 * CM_PER_S; *dp++ = 1.224224e-09 * G_PER_CM3; n++;
    *xp++ = 3.970000e-02; *vtp++ = 1.495520e+07 * CM_PER_S; *vrp++ = 5.783410e+02 * CM_PER_S; *dp++ = 1.217765e-09 * G_PER_CM3; n++;
    *xp++ = 4.085000e-02; *vtp++ = 1.474319e+07 * CM_PER_S; *vrp++ = 5.437512e+02 * CM_PER_S; *dp++ = 1.211658e-09 * G_PER_CM3; n++;
    *xp++ = 4.203400e-02; *vtp++ = 1.453414e+07 * CM_PER_S; *vrp++ = 5.107827e+02 * CM_PER_S; *dp++ = 1.206608e-09 * G_PER_CM3; n++;
    *xp++ = 4.325200e-02; *vtp++ = 1.432800e+07 * CM_PER_S; *vrp++ = 4.793798e+02 * CM_PER_S; *dp++ = 1.202671e-09 * G_PER_CM3; n++;
    *xp++ = 4.450600e-02; *vtp++ = 1.412492e+07 * CM_PER_S; *vrp++ = 4.494893e+02 * CM_PER_S; *dp++ = 1.199817e-09 * G_PER_CM3; n++;
    *xp++ = 4.579500e-02; *vtp++ = 1.392458e+07 * CM_PER_S; *vrp++ = 4.210602e+02 * CM_PER_S; *dp++ = 1.198146e-09 * G_PER_CM3; n++;
    *xp++ = 4.712200e-02; *vtp++ = 1.372723e+07 * CM_PER_S; *vrp++ = 3.940433e+02 * CM_PER_S; *dp++ = 1.197592e-09 * G_PER_CM3; n++;
    *xp++ = 4.848800e-02; *vtp++ = 1.353221e+07 * CM_PER_S; *vrp++ = 3.683908e+02 * CM_PER_S; *dp++ = 1.198184e-09 * G_PER_CM3; n++;
    *xp++ = 4.989300e-02; *vtp++ = 1.334318e+07 * CM_PER_S; *vrp++ = 3.458213e+02 * CM_PER_S; *dp++ = 1.194622e-09 * G_PER_CM3; n++;
    *xp++ = 5.133900e-02; *vtp++ = 1.315395e+07 * CM_PER_S; *vrp++ = 3.129678e+02 * CM_PER_S; *dp++ = 1.229755e-09 * G_PER_CM3; n++;
    *xp++ = 5.282700e-02; *vtp++ = 1.296745e+07 * CM_PER_S; *vrp++ = 2.835680e+02 * CM_PER_S; *dp++ = 1.263817e-09 * G_PER_CM3; n++;
    *xp++ = 5.435800e-02; *vtp++ = 1.278359e+07 * CM_PER_S; *vrp++ = 2.569423e+02 * CM_PER_S; *dp++ = 1.298845e-09 * G_PER_CM3; n++;
    *xp++ = 5.593300e-02; *vtp++ = 1.260234e+07 * CM_PER_S; *vrp++ = 2.328252e+02 * CM_PER_S; *dp++ = 1.334889e-09 * G_PER_CM3; n++;
    *xp++ = 5.755400e-02; *vtp++ = 1.242366e+07 * CM_PER_S; *vrp++ = 2.109773e+02 * CM_PER_S; *dp++ = 1.371961e-09 * G_PER_CM3; n++;
    *xp++ = 5.922200e-02; *vtp++ = 1.224752e+07 * CM_PER_S; *vrp++ = 1.911825e+02 * CM_PER_S; *dp++ = 1.410113e-09 * G_PER_CM3; n++;
    *xp++ = 6.093800e-02; *vtp++ = 1.207388e+07 * CM_PER_S; *vrp++ = 1.732460e+02 * CM_PER_S; *dp++ = 1.449402e-09 * G_PER_CM3; n++;
    *xp++ = 6.270400e-02; *vtp++ = 1.190270e+07 * CM_PER_S; *vrp++ = 1.569918e+02 * CM_PER_S; *dp++ = 1.489849e-09 * G_PER_CM3; n++;
    *xp++ = 6.452100e-02; *vtp++ = 1.173395e+07 * CM_PER_S; *vrp++ = 1.422609e+02 * CM_PER_S; *dp++ = 1.531517e-09 * G_PER_CM3; n++;
    *xp++ = 6.639100e-02; *vtp++ = 1.156760e+07 * CM_PER_S; *vrp++ = 1.289093e+02 * CM_PER_S; *dp++ = 1.574441e-09 * G_PER_CM3; n++;
    *xp++ = 6.831500e-02; *vtp++ = 1.140361e+07 * CM_PER_S; *vrp++ = 1.168000e+02 * CM_PER_S; *dp++ = 1.618840e-09 * G_PER_CM3; n++;
    *xp++ = 7.029400e-02; *vtp++ = 1.124194e+07 * CM_PER_S; *vrp++ = 1.058243e+02 * CM_PER_S; *dp++ = 1.664641e-09 * G_PER_CM3; n++;
    *xp++ = 7.233100e-02; *vtp++ = 1.108257e+07 * CM_PER_S; *vrp++ = 9.587575e+01 * CM_PER_S; *dp++ = 1.711851e-09 * G_PER_CM3; n++;
    *xp++ = 7.442800e-02; *vtp++ = 1.092546e+07 * CM_PER_S; *vrp++ = 8.685811e+01 * CM_PER_S; *dp++ = 1.760511e-09 * G_PER_CM3; n++;
    *xp++ = 7.658400e-02; *vtp++ = 1.077057e+07 * CM_PER_S; *vrp++ = 7.868416e+01 * CM_PER_S; *dp++ = 1.810778e-09 * G_PER_CM3; n++;
    *xp++ = 7.880400e-02; *vtp++ = 1.061789e+07 * CM_PER_S; *vrp++ = 7.127498e+01 * CM_PER_S; *dp++ = 1.862596e-09 * G_PER_CM3; n++;
    *xp++ = 8.108700e-02; *vtp++ = 1.046737e+07 * CM_PER_S; *vrp++ = 6.455905e+01 * CM_PER_S; *dp++ = 1.916130e-09 * G_PER_CM3; n++;
    *xp++ = 8.343700e-02; *vtp++ = 1.031899e+07 * CM_PER_S; *vrp++ = 5.847161e+01 * CM_PER_S; *dp++ = 1.971370e-09 * G_PER_CM3; n++;
    *xp++ = 8.585500e-02; *vtp++ = 1.017271e+07 * CM_PER_S; *vrp++ = 5.295399e+01 * CM_PER_S; *dp++ = 2.028415e-09 * G_PER_CM3; n++;
    *xp++ = 8.834300e-02; *vtp++ = 1.002852e+07 * CM_PER_S; *vrp++ = 4.795347e+01 * CM_PER_S; *dp++ = 2.087305e-09 * G_PER_CM3; n++;
    *xp++ = 9.090300e-02; *vtp++ = 9.886367e+06 * CM_PER_S; *vrp++ = 4.342135e+01 * CM_PER_S; *dp++ = 2.148142e-09 * G_PER_CM3; n++;
    *xp++ = 9.353800e-02; *vtp++ = 9.746232e+06 * CM_PER_S; *vrp++ = 3.931390e+01 * CM_PER_S; *dp++ = 2.210975e-09 * G_PER_CM3; n++;
    *xp++ = 9.624800e-02; *vtp++ = 9.608086e+06 * CM_PER_S; *vrp++ = 3.559153e+01 * CM_PER_S; *dp++ = 2.275960e-09 * G_PER_CM3; n++;
    *xp++ = 9.903800e-02; *vtp++ = 9.471901e+06 * CM_PER_S; *vrp++ = 3.221833e+01 * CM_PER_S; *dp++ = 2.343080e-09 * G_PER_CM3; n++;
    *xp++ = 1.019080e-01; *vtp++ = 9.337649e+06 * CM_PER_S; *vrp++ = 2.916174e+01 * CM_PER_S; *dp++ = 2.412507e-09 * G_PER_CM3; n++;
    *xp++ = 1.048610e-01; *vtp++ = 9.205303e+06 * CM_PER_S; *vrp++ = 2.639224e+01 * CM_PER_S; *dp++ = 2.484307e-09 * G_PER_CM3; n++;
    *xp++ = 1.079000e-01; *vtp++ = 9.074835e+06 * CM_PER_S; *vrp++ = 2.388306e+01 * CM_PER_S; *dp++ = 2.558556e-09 * G_PER_CM3; n++;
    *xp++ = 1.110260e-01; *vtp++ = 8.946217e+06 * CM_PER_S; *vrp++ = 2.160990e+01 * CM_PER_S; *dp++ = 2.635405e-09 * G_PER_CM3; n++;
    *xp++ = 1.142440e-01; *vtp++ = 8.819426e+06 * CM_PER_S; *vrp++ = 1.955160e+01 * CM_PER_S; *dp++ = 2.714694e-09 * G_PER_CM3; n++;
    *xp++ = 1.175550e-01; *vtp++ = 8.694436e+06 * CM_PER_S; *vrp++ = 1.768740e+01 * CM_PER_S; *dp++ = 2.796700e-09 * G_PER_CM3; n++;
    *xp++ = 1.209610e-01; *vtp++ = 8.571220e+06 * CM_PER_S; *vrp++ = 1.599894e+01 * CM_PER_S; *dp++ = 2.881606e-09 * G_PER_CM3; n++;
    *xp++ = 1.244670e-01; *vtp++ = 8.449754e+06 * CM_PER_S; *vrp++ = 1.446981e+01 * CM_PER_S; *dp++ = 2.969452e-09 * G_PER_CM3; n++;
    *xp++ = 1.280740e-01; *vtp++ = 8.330011e+06 * CM_PER_S; *vrp++ = 1.308511e+01 * CM_PER_S; *dp++ = 3.060425e-09 * G_PER_CM3; n++;
    *xp++ = 1.317850e-01; *vtp++ = 8.211969e+06 * CM_PER_S; *vrp++ = 1.183133e+01 * CM_PER_S; *dp++ = 3.154651e-09 * G_PER_CM3; n++;
    *xp++ = 1.356040e-01; *vtp++ = 8.095603e+06 * CM_PER_S; *vrp++ = 1.069623e+01 * CM_PER_S; *dp++ = 3.252230e-09 * G_PER_CM3; n++;
    *xp++ = 1.395340e-01; *vtp++ = 7.980889e+06 * CM_PER_S; *vrp++ = 9.668670e+00 * CM_PER_S; *dp++ = 3.353304e-09 * G_PER_CM3; n++;
    *xp++ = 1.435770e-01; *vtp++ = 7.867802e+06 * CM_PER_S; *vrp++ = 8.738585e+00 * CM_PER_S; *dp++ = 3.458062e-09 * G_PER_CM3; n++;
    *xp++ = 1.477380e-01; *vtp++ = 7.756306e+06 * CM_PER_S; *vrp++ = 7.897191e+00 * CM_PER_S; *dp++ = 3.566352e-09 * G_PER_CM3; n++;
    *xp++ = 1.520190e-01; *vtp++ = 7.646425e+06 * CM_PER_S; *vrp++ = 7.135946e+00 * CM_PER_S; *dp++ = 3.678467e-09 * G_PER_CM3; n++;
    *xp++ = 1.564250e-01; *vtp++ = 7.538089e+06 * CM_PER_S; *vrp++ = 6.447131e+00 * CM_PER_S; *dp++ = 3.794642e-09 * G_PER_CM3; n++;
    *xp++ = 1.609580e-01; *vtp++ = 7.431292e+06 * CM_PER_S; *vrp++ = 5.823934e+00 * CM_PER_S; *dp++ = 3.915106e-09 * G_PER_CM3; n++;
    *xp++ = 1.656220e-01; *vtp++ = 7.326012e+06 * CM_PER_S; *vrp++ = 5.260177e+00 * CM_PER_S; *dp++ = 4.040031e-09 * G_PER_CM3; n++;
    *xp++ = 1.704220e-01; *vtp++ = 7.222226e+06 * CM_PER_S; *vrp++ = 4.750260e+00 * CM_PER_S; *dp++ = 4.169556e-09 * G_PER_CM3; n++;
    *xp++ = 1.753610e-01; *vtp++ = 7.119915e+06 * CM_PER_S; *vrp++ = 4.289104e+00 * CM_PER_S; *dp++ = 4.303915e-09 * G_PER_CM3; n++;
    *xp++ = 1.804420e-01; *vtp++ = 7.019058e+06 * CM_PER_S; *vrp++ = 3.872105e+00 * CM_PER_S; *dp++ = 4.443351e-09 * G_PER_CM3; n++;
    *xp++ = 1.856720e-01; *vtp++ = 6.919651e+06 * CM_PER_S; *vrp++ = 3.495087e+00 * CM_PER_S; *dp++ = 4.587972e-09 * G_PER_CM3; n++;
    *xp++ = 1.910520e-01; *vtp++ = 6.821656e+06 * CM_PER_S; *vrp++ = 3.153323e+00 * CM_PER_S; *dp++ = 4.740253e-09 * G_PER_CM3; n++;
    *xp++ = 1.965890e-01; *vtp++ = 6.725035e+06 * CM_PER_S; *vrp++ = 2.843694e+00 * CM_PER_S; *dp++ = 4.900466e-09 * G_PER_CM3; n++;
    *xp++ = 2.022860e-01; *vtp++ = 6.629787e+06 * CM_PER_S; *vrp++ = 2.563988e+00 * CM_PER_S; *dp++ = 5.067103e-09 * G_PER_CM3; n++;
    *xp++ = 2.081480e-01; *vtp++ = 6.535891e+06 * CM_PER_S; *vrp++ = 2.311354e+00 * CM_PER_S; *dp++ = 5.240458e-09 * G_PER_CM3; n++;
    *xp++ = 2.141800e-01; *vtp++ = 6.443330e+06 * CM_PER_S; *vrp++ = 2.083212e+00 * CM_PER_S; *dp++ = 5.420834e-09 * G_PER_CM3; n++;
    *xp++ = 2.203860e-01; *vtp++ = 6.352084e+06 * CM_PER_S; *vrp++ = 1.877222e+00 * CM_PER_S; *dp++ = 5.608605e-09 * G_PER_CM3; n++;
    *xp++ = 2.267730e-01; *vtp++ = 6.258068e+06 * CM_PER_S; *vrp++ = 1.691265e+00 * CM_PER_S; *dp++ = 5.804042e-09 * G_PER_CM3; n++;
    *xp++ = 2.333450e-01; *vtp++ = 6.159572e+06 * CM_PER_S; *vrp++ = 1.564110e+00 * CM_PER_S; *dp++ = 5.870605e-09 * G_PER_CM3; n++;
    *xp++ = 2.401070e-01; *vtp++ = 6.072233e+06 * CM_PER_S; *vrp++ = 1.540226e+00 * CM_PER_S; *dp++ = 5.620804e-09 * G_PER_CM3; n++;
    *xp++ = 2.470650e-01; *vtp++ = 5.986177e+06 * CM_PER_S; *vrp++ = 1.515473e+00 * CM_PER_S; *dp++ = 5.388429e-09 * G_PER_CM3; n++;
    *xp++ = 2.542250e-01; *vtp++ = 5.901272e+06 * CM_PER_S; *vrp++ = 1.489131e+00 * CM_PER_S; *dp++ = 5.176211e-09 * G_PER_CM3; n++;
    *xp++ = 2.615920e-01; *vtp++ = 5.817574e+06 * CM_PER_S; *vrp++ = 1.463078e+00 * CM_PER_S; *dp++ = 4.973426e-09 * G_PER_CM3; n++;
    *xp++ = 2.691730e-01; *vtp++ = 5.735066e+06 * CM_PER_S; *vrp++ = 1.437298e+00 * CM_PER_S; *dp++ = 4.779673e-09 * G_PER_CM3; n++;
    *xp++ = 2.769730e-01; *vtp++ = 5.653731e+06 * CM_PER_S; *vrp++ = 1.411774e+00 * CM_PER_S; *dp++ = 4.594633e-09 * G_PER_CM3; n++;
    *xp++ = 2.850000e-01; *vtp++ = 5.573553e+06 * CM_PER_S; *vrp++ = 1.386489e+00 * CM_PER_S; *dp++ = 4.417926e-09 * G_PER_CM3; n++;
    *xp++ = 2.932590e-01; *vtp++ = 5.494515e+06 * CM_PER_S; *vrp++ = 1.361426e+00 * CM_PER_S; *dp++ = 4.249271e-09 * G_PER_CM3; n++;
    *xp++ = 3.017570e-01; *vtp++ = 5.416593e+06 * CM_PER_S; *vrp++ = 1.336564e+00 * CM_PER_S; *dp++ = 4.088360e-09 * G_PER_CM3; n++;
    *xp++ = 3.105020e-01; *vtp++ = 5.339765e+06 * CM_PER_S; *vrp++ = 1.311886e+00 * CM_PER_S; *dp++ = 3.934886e-09 * G_PER_CM3; n++;
    *xp++ = 3.195000e-01; *vtp++ = 5.263962e+06 * CM_PER_S; *vrp++ = 1.288013e+00 * CM_PER_S; *dp++ = 3.785775e-09 * G_PER_CM3; n++;
    *xp++ = 3.287590e-01; *vtp++ = 5.189304e+06 * CM_PER_S; *vrp++ = 1.266118e+00 * CM_PER_S; *dp++ = 3.635770e-09 * G_PER_CM3; n++;
    *xp++ = 3.382860e-01; *vtp++ = 5.115708e+06 * CM_PER_S; *vrp++ = 1.244441e+00 * CM_PER_S; *dp++ = 3.492486e-09 * G_PER_CM3; n++;
    *xp++ = 3.480890e-01; *vtp++ = 5.043157e+06 * CM_PER_S; *vrp++ = 1.222967e+00 * CM_PER_S; *dp++ = 3.355646e-09 * G_PER_CM3; n++;
    *xp++ = 3.581770e-01; *vtp++ = 4.971638e+06 * CM_PER_S; *vrp++ = 1.201684e+00 * CM_PER_S; *dp++ = 3.224981e-09 * G_PER_CM3; n++;
    *xp++ = 3.685560e-01; *vtp++ = 4.901130e+06 * CM_PER_S; *vrp++ = 1.180578e+00 * CM_PER_S; *dp++ = 3.100283e-09 * G_PER_CM3; n++;
    *xp++ = 3.792370e-01; *vtp++ = 4.831637e+06 * CM_PER_S; *vrp++ = 1.159634e+00 * CM_PER_S; *dp++ = 2.981283e-09 * G_PER_CM3; n++;
    *xp++ = 3.902270e-01; *vtp++ = 4.763126e+06 * CM_PER_S; *vrp++ = 1.138838e+00 * CM_PER_S; *dp++ = 2.867789e-09 * G_PER_CM3; n++;
    *xp++ = 4.015350e-01; *vtp++ = 4.695575e+06 * CM_PER_S; *vrp++ = 1.118174e+00 * CM_PER_S; *dp++ = 2.759592e-09 * G_PER_CM3; n++;
    *xp++ = 4.131710e-01; *vtp++ = 4.628908e+06 * CM_PER_S; *vrp++ = 1.097910e+00 * CM_PER_S; *dp++ = 2.655461e-09 * G_PER_CM3; n++;
    *xp++ = 4.251450e-01; *vtp++ = 4.563260e+06 * CM_PER_S; *vrp++ = 1.079597e+00 * CM_PER_S; *dp++ = 2.549701e-09 * G_PER_CM3; n++;
    *xp++ = 4.374650e-01; *vtp++ = 4.498544e+06 * CM_PER_S; *vrp++ = 1.061450e+00 * CM_PER_S; *dp++ = 2.448714e-09 * G_PER_CM3; n++;
    *xp++ = 4.501430e-01; *vtp++ = 4.434749e+06 * CM_PER_S; *vrp++ = 1.043458e+00 * CM_PER_S; *dp++ = 2.352291e-09 * G_PER_CM3; n++;
    *xp++ = 4.631870e-01; *vtp++ = 4.371860e+06 * CM_PER_S; *vrp++ = 1.025609e+00 * CM_PER_S; *dp++ = 2.260267e-09 * G_PER_CM3; n++;
    *xp++ = 4.766100e-01; *vtp++ = 4.309866e+06 * CM_PER_S; *vrp++ = 1.007894e+00 * CM_PER_S; *dp++ = 2.172448e-09 * G_PER_CM3; n++;
    *xp++ = 4.904220e-01; *vtp++ = 4.248753e+06 * CM_PER_S; *vrp++ = 9.903004e-01 * CM_PER_S; *dp++ = 2.088677e-09 * G_PER_CM3; n++;
    *xp++ = 5.046340e-01; *vtp++ = 4.188509e+06 * CM_PER_S; *vrp++ = 9.728161e-01 * CM_PER_S; *dp++ = 2.008801e-09 * G_PER_CM3; n++;
    *xp++ = 5.192580e-01; *vtp++ = 4.129122e+06 * CM_PER_S; *vrp++ = 9.554283e-01 * CM_PER_S; *dp++ = 1.932670e-09 * G_PER_CM3; n++;
    *xp++ = 5.343060e-01; *vtp++ = 4.070488e+06 * CM_PER_S; *vrp++ = 9.381345e-01 * CM_PER_S; *dp++ = 1.860112e-09 * G_PER_CM3; n++;
    *xp++ = 5.497890e-01; *vtp++ = 4.012761e+06 * CM_PER_S; *vrp++ = 9.227391e-01 * CM_PER_S; *dp++ = 1.785713e-09 * G_PER_CM3; n++;
    *xp++ = 5.657220e-01; *vtp++ = 3.955855e+06 * CM_PER_S; *vrp++ = 9.074714e-01 * CM_PER_S; *dp++ = 1.714676e-09 * G_PER_CM3; n++;
    *xp++ = 5.821160e-01; *vtp++ = 3.899757e+06 * CM_PER_S; *vrp++ = 8.923230e-01 * CM_PER_S; *dp++ = 1.646876e-09 * G_PER_CM3; n++;
    *xp++ = 5.989850e-01; *vtp++ = 3.844457e+06 * CM_PER_S; *vrp++ = 8.772852e-01 * CM_PER_S; *dp++ = 1.582177e-09 * G_PER_CM3; n++;
    *xp++ = 6.163430e-01; *vtp++ = 3.789943e+06 * CM_PER_S; *vrp++ = 8.623491e-01 * CM_PER_S; *dp++ = 1.520453e-09 * G_PER_CM3; n++;
    *xp++ = 6.342050e-01; *vtp++ = 3.736204e+06 * CM_PER_S; *vrp++ = 8.475050e-01 * CM_PER_S; *dp++ = 1.461584e-09 * G_PER_CM3; n++;
    *xp++ = 6.525830e-01; *vtp++ = 3.683229e+06 * CM_PER_S; *vrp++ = 8.327429e-01 * CM_PER_S; *dp++ = 1.405467e-09 * G_PER_CM3; n++;
    *xp++ = 6.714950e-01; *vtp++ = 3.631008e+06 * CM_PER_S; *vrp++ = 8.180524e-01 * CM_PER_S; *dp++ = 1.351989e-09 * G_PER_CM3; n++;
    *xp++ = 6.909540e-01; *vtp++ = 3.579458e+06 * CM_PER_S; *vrp++ = 8.034218e-01 * CM_PER_S; *dp++ = 1.301059e-09 * G_PER_CM3; n++;
    *xp++ = 7.109770e-01; *vtp++ = 3.528688e+06 * CM_PER_S; *vrp++ = 7.902319e-01 * CM_PER_S; *dp++ = 1.249270e-09 * G_PER_CM3; n++;
    *xp++ = 7.315810e-01; *vtp++ = 3.478648e+06 * CM_PER_S; *vrp++ = 7.773340e-01 * CM_PER_S; *dp++ = 1.199391e-09 * G_PER_CM3; n++;
    *xp++ = 7.527820e-01; *vtp++ = 3.429318e+06 * CM_PER_S; *vrp++ = 7.645291e-01 * CM_PER_S; *dp++ = 1.151790e-09 * G_PER_CM3; n++;
    *xp++ = 7.745970e-01; *vtp++ = 3.380690e+06 * CM_PER_S; *vrp++ = 7.518103e-01 * CM_PER_S; *dp++ = 1.106374e-09 * G_PER_CM3; n++;
    *xp++ = 7.970440e-01; *vtp++ = 3.332753e+06 * CM_PER_S; *vrp++ = 7.391700e-01 * CM_PER_S; *dp++ = 1.063057e-09 * G_PER_CM3; n++;
    *xp++ = 8.201420e-01; *vtp++ = 3.285498e+06 * CM_PER_S; *vrp++ = 7.266004e-01 * CM_PER_S; *dp++ = 1.021752e-09 * G_PER_CM3; n++;
    *xp++ = 8.439090e-01; *vtp++ = 3.238915e+06 * CM_PER_S; *vrp++ = 7.140932e-01 * CM_PER_S; *dp++ = 9.823815e-10 * G_PER_CM3; n++;
    *xp++ = 8.683640e-01; *vtp++ = 3.192994e+06 * CM_PER_S; *vrp++ = 7.016396e-01 * CM_PER_S; *dp++ = 9.448729e-10 * G_PER_CM3; n++;
    *xp++ = 8.935290e-01; *vtp++ = 3.147673e+06 * CM_PER_S; *vrp++ = 6.892300e-01 * CM_PER_S; *dp++ = 9.091515e-10 * G_PER_CM3; n++;
    *xp++ = 9.194230e-01; *vtp++ = 3.103021e+06 * CM_PER_S; *vrp++ = 6.778779e-01 * CM_PER_S; *dp++ = 8.731743e-10 * G_PER_CM3; n++;
    *xp++ = 9.460670e-01; *vtp++ = 3.059017e+06 * CM_PER_S; *vrp++ = 6.669444e-01 * CM_PER_S; *dp++ = 8.381984e-10 * G_PER_CM3; n++;
    *xp++ = 9.734830e-01; *vtp++ = 3.015640e+06 * CM_PER_S; *vrp++ = 6.560845e-01 * CM_PER_S; *dp++ = 8.048254e-10 * G_PER_CM3; n++;
    *xp++ = 1.001694e+00; *vtp++ = 2.972878e+06 * CM_PER_S; *vrp++ = 6.452922e-01 * CM_PER_S; *dp++ = 7.729893e-10 * G_PER_CM3; n++;
    *xp++ = 1.030722e+00; *vtp++ = 2.930725e+06 * CM_PER_S; *vrp++ = 6.345614e-01 * CM_PER_S; *dp++ = 7.426278e-10 * G_PER_CM3; n++;
    *xp++ = 1.060592e+00; *vtp++ = 2.889170e+06 * CM_PER_S; *vrp++ = 6.238856e-01 * CM_PER_S; *dp++ = 7.136806e-10 * G_PER_CM3; n++;
    *xp++ = 1.091327e+00; *vtp++ = 2.848207e+06 * CM_PER_S; *vrp++ = 6.132578e-01 * CM_PER_S; *dp++ = 6.860938e-10 * G_PER_CM3; n++;
    *xp++ = 1.122953e+00; *vtp++ = 2.807826e+06 * CM_PER_S; *vrp++ = 6.026706e-01 * CM_PER_S; *dp++ = 6.598129e-10 * G_PER_CM3; n++;
    *xp++ = 1.155495e+00; *vtp++ = 2.767981e+06 * CM_PER_S; *vrp++ = 5.921160e-01 * CM_PER_S; *dp++ = 6.347899e-10 * G_PER_CM3; n++;
    *xp++ = 1.188980e+00; *vtp++ = 2.728709e+06 * CM_PER_S; *vrp++ = 5.823150e-01 * CM_PER_S; *dp++ = 6.098301e-10 * G_PER_CM3; n++;
    *xp++ = 1.223436e+00; *vtp++ = 2.690015e+06 * CM_PER_S; *vrp++ = 5.730195e-01 * CM_PER_S; *dp++ = 5.853327e-10 * G_PER_CM3; n++;
    *xp++ = 1.258890e+00; *vtp++ = 2.651870e+06 * CM_PER_S; *vrp++ = 5.637828e-01 * CM_PER_S; *dp++ = 5.619615e-10 * G_PER_CM3; n++;
    *xp++ = 1.295372e+00; *vtp++ = 2.614267e+06 * CM_PER_S; *vrp++ = 5.545999e-01 * CM_PER_S; *dp++ = 5.396690e-10 * G_PER_CM3; n++;
    *xp++ = 1.332911e+00; *vtp++ = 2.577199e+06 * CM_PER_S; *vrp++ = 5.454657e-01 * CM_PER_S; *dp++ = 5.184117e-10 * G_PER_CM3; n++;
    *xp++ = 1.371538e+00; *vtp++ = 2.540658e+06 * CM_PER_S; *vrp++ = 5.363746e-01 * CM_PER_S; *dp++ = 4.981475e-10 * G_PER_CM3; n++;
    *xp++ = 1.411284e+00; *vtp++ = 2.504636e+06 * CM_PER_S; *vrp++ = 5.273209e-01 * CM_PER_S; *dp++ = 4.788371e-10 * G_PER_CM3; n++;
    *xp++ = 1.452182e+00; *vtp++ = 2.469127e+06 * CM_PER_S; *vrp++ = 5.182983e-01 * CM_PER_S; *dp++ = 4.604430e-10 * G_PER_CM3; n++;
    *xp++ = 1.494265e+00; *vtp++ = 2.434074e+06 * CM_PER_S; *vrp++ = 5.093002e-01 * CM_PER_S; *dp++ = 4.429307e-10 * G_PER_CM3; n++;
    *xp++ = 1.537567e+00; *vtp++ = 2.399506e+06 * CM_PER_S; *vrp++ = 5.012277e-01 * CM_PER_S; *dp++ = 4.251089e-10 * G_PER_CM3; n++;
    *xp++ = 1.582125e+00; *vtp++ = 2.365138e+06 * CM_PER_S; *vrp++ = 4.941634e-01 * CM_PER_S; *dp++ = 4.069189e-10 * G_PER_CM3; n++;
    *xp++ = 1.627974e+00; *vtp++ = 2.331483e+06 * CM_PER_S; *vrp++ = 4.934257e-01 * CM_PER_S; *dp++ = 3.821617e-10 * G_PER_CM3; n++;
    *xp++ = 1.675151e+00; *vtp++ = 2.298350e+06 * CM_PER_S; *vrp++ = 4.937192e-01 * CM_PER_S; *dp++ = 3.577924e-10 * G_PER_CM3; n++;
    *xp++ = 1.723696e+00; *vtp++ = 2.265687e+06 * CM_PER_S; *vrp++ = 4.940241e-01 * CM_PER_S; *dp++ = 3.349692e-10 * G_PER_CM3; n++;
    *xp++ = 1.773647e+00; *vtp++ = 2.233487e+06 * CM_PER_S; *vrp++ = 4.943401e-01 * CM_PER_S; *dp++ = 3.135951e-10 * G_PER_CM3; n++;
    *xp++ = 1.825046e+00; *vtp++ = 2.201743e+06 * CM_PER_S; *vrp++ = 4.946671e-01 * CM_PER_S; *dp++ = 2.935784e-10 * G_PER_CM3; n++;
    *xp++ = 1.877934e+00; *vtp++ = 2.170450e+06 * CM_PER_S; *vrp++ = 4.950049e-01 * CM_PER_S; *dp++ = 2.748335e-10 * G_PER_CM3; n++;
    *xp++ = 1.932356e+00; *vtp++ = 2.139600e+06 * CM_PER_S; *vrp++ = 4.953534e-01 * CM_PER_S; *dp++ = 2.572800e-10 * G_PER_CM3; n++;
    *xp++ = 1.988354e+00; *vtp++ = 2.109187e+06 * CM_PER_S; *vrp++ = 4.957124e-01 * CM_PER_S; *dp++ = 2.408426e-10 * G_PER_CM3; n++;
    *xp++ = 2.045975e+00; *vtp++ = 2.079206e+06 * CM_PER_S; *vrp++ = 4.960817e-01 * CM_PER_S; *dp++ = 2.254509e-10 * G_PER_CM3; n++;
    *xp++ = 2.105265e+00; *vtp++ = 2.049650e+06 * CM_PER_S; *vrp++ = 4.964613e-01 * CM_PER_S; *dp++ = 2.110387e-10 * G_PER_CM3; n++;
    *xp++ = 2.166274e+00; *vtp++ = 2.020513e+06 * CM_PER_S; *vrp++ = 4.968509e-01 * CM_PER_S; *dp++ = 1.975438e-10 * G_PER_CM3; n++;
    *xp++ = 2.229051e+00; *vtp++ = 1.991789e+06 * CM_PER_S; *vrp++ = 4.972504e-01 * CM_PER_S; *dp++ = 1.849083e-10 * G_PER_CM3; n++;
    *xp++ = 2.293647e+00; *vtp++ = 1.963472e+06 * CM_PER_S; *vrp++ = 4.976598e-01 * CM_PER_S; *dp++ = 1.730777e-10 * G_PER_CM3; n++;
    *xp++ = 2.360116e+00; *vtp++ = 1.935557e+06 * CM_PER_S; *vrp++ = 4.980788e-01 * CM_PER_S; *dp++ = 1.620009e-10 * G_PER_CM3; n++;
    *xp++ = 2.428510e+00; *vtp++ = 1.908037e+06 * CM_PER_S; *vrp++ = 4.985073e-01 * CM_PER_S; *dp++ = 1.516303e-10 * G_PER_CM3; n++;
    *xp++ = 2.498886e+00; *vtp++ = 1.880907e+06 * CM_PER_S; *vrp++ = 4.989452e-01 * CM_PER_S; *dp++ = 1.419210e-10 * G_PER_CM3; n++;
    *xp++ = 2.571302e+00; *vtp++ = 1.854163e+06 * CM_PER_S; *vrp++ = 4.993925e-01 * CM_PER_S; *dp++ = 1.328311e-10 * G_PER_CM3; n++;
    *xp++ = 2.645816e+00; *vtp++ = 1.827797e+06 * CM_PER_S; *vrp++ = 4.998488e-01 * CM_PER_S; *dp++ = 1.243212e-10 * G_PER_CM3; n++;
    *xp++ = 2.722490e+00; *vtp++ = 1.801805e+06 * CM_PER_S; *vrp++ = 5.003142e-01 * CM_PER_S; *dp++ = 1.163545e-10 * G_PER_CM3; n++;
    *xp++ = 2.801386e+00; *vtp++ = 1.776181e+06 * CM_PER_S; *vrp++ = 5.007886e-01 * CM_PER_S; *dp++ = 1.088963e-10 * G_PER_CM3; n++;
    *xp++ = 2.882568e+00; *vtp++ = 1.750921e+06 * CM_PER_S; *vrp++ = 5.012717e-01 * CM_PER_S; *dp++ = 1.019146e-10 * G_PER_CM3; n++;
    *xp++ = 2.966102e+00; *vtp++ = 1.726019e+06 * CM_PER_S; *vrp++ = 5.017635e-01 * CM_PER_S; *dp++ = 9.537904e-11 * G_PER_CM3; n++;
    *xp++ = 3.052058e+00; *vtp++ = 1.701470e+06 * CM_PER_S; *vrp++ = 5.022639e-01 * CM_PER_S; *dp++ = 8.926102e-11 * G_PER_CM3; n++;
    *xp++ = 3.140504e+00; *vtp++ = 1.677269e+06 * CM_PER_S; *vrp++ = 5.027728e-01 * CM_PER_S; *dp++ = 8.353420e-11 * G_PER_CM3; n++;
    *xp++ = 3.231513e+00; *vtp++ = 1.653411e+06 * CM_PER_S; *vrp++ = 5.032900e-01 * CM_PER_S; *dp++ = 7.817352e-11 * G_PER_CM3; n++;
    *xp++ = 3.325160e+00; *vtp++ = 1.629891e+06 * CM_PER_S; *vrp++ = 5.038156e-01 * CM_PER_S; *dp++ = 7.315576e-11 * G_PER_CM3; n++;
    *xp++ = 3.421521e+00; *vtp++ = 1.606705e+06 * CM_PER_S; *vrp++ = 5.043493e-01 * CM_PER_S; *dp++ = 6.845900e-11 * G_PER_CM3; n++;
    *xp++ = 3.520674e+00; *vtp++ = 1.583847e+06 * CM_PER_S; *vrp++ = 5.048910e-01 * CM_PER_S; *dp++ = 6.406289e-11 * G_PER_CM3; n++;
    *xp++ = 3.622700e+00; *vtp++ = 1.561313e+06 * CM_PER_S; *vrp++ = 5.054408e-01 * CM_PER_S; *dp++ = 5.994817e-11 * G_PER_CM3; n++;
    *xp++ = 3.727683e+00; *vtp++ = 1.539099e+06 * CM_PER_S; *vrp++ = 5.059984e-01 * CM_PER_S; *dp++ = 5.609693e-11 * G_PER_CM3; n++;
    *xp++ = 3.835708e+00; *vtp++ = 1.517199e+06 * CM_PER_S; *vrp++ = 5.065638e-01 * CM_PER_S; *dp++ = 5.249237e-11 * G_PER_CM3; n++;
    *xp++ = 3.946864e+00; *vtp++ = 1.495611e+06 * CM_PER_S; *vrp++ = 5.071369e-01 * CM_PER_S; *dp++ = 4.911872e-11 * G_PER_CM3; n++;
    *xp++ = 4.061241e+00; *vtp++ = 1.474328e+06 * CM_PER_S; *vrp++ = 5.077176e-01 * CM_PER_S; *dp++ = 4.596128e-11 * G_PER_CM3; n++;
    *xp++ = 4.178933e+00; *vtp++ = 1.453346e+06 * CM_PER_S; *vrp++ = 5.083059e-01 * CM_PER_S; *dp++ = 4.300622e-11 * G_PER_CM3; n++;
    *xp++ = 4.300035e+00; *vtp++ = 1.432663e+06 * CM_PER_S; *vrp++ = 5.089015e-01 * CM_PER_S; *dp++ = 4.024063e-11 * G_PER_CM3; n++;
    *xp++ = 4.424647e+00; *vtp++ = 1.412272e+06 * CM_PER_S; *vrp++ = 5.095046e-01 * CM_PER_S; *dp++ = 3.765238e-11 * G_PER_CM3; n++;
    *xp++ = 4.552870e+00; *vtp++ = 1.392171e+06 * CM_PER_S; *vrp++ = 5.101149e-01 * CM_PER_S; *dp++ = 3.523015e-11 * G_PER_CM3; n++;
    *xp++ = 4.684808e+00; *vtp++ = 1.372354e+06 * CM_PER_S; *vrp++ = 5.107324e-01 * CM_PER_S; *dp++ = 3.296334e-11 * G_PER_CM3; n++;
    *xp++ = 4.820571e+00; *vtp++ = 1.352818e+06 * CM_PER_S; *vrp++ = 5.113570e-01 * CM_PER_S; *dp++ = 3.084199e-11 * G_PER_CM3; n++;
    *xp++ = 4.960267e+00; *vtp++ = 1.333559e+06 * CM_PER_S; *vrp++ = 5.119887e-01 * CM_PER_S; *dp++ = 2.885681e-11 * G_PER_CM3; n++;
    *xp++ = 5.104012e+00; *vtp++ = 1.314573e+06 * CM_PER_S; *vrp++ = 5.126273e-01 * CM_PER_S; *dp++ = 2.699908e-11 * G_PER_CM3; n++;
    *xp++ = 5.251922e+00; *vtp++ = 1.295856e+06 * CM_PER_S; *vrp++ = 5.132728e-01 * CM_PER_S; *dp++ = 2.526065e-11 * G_PER_CM3; n++;
    *xp++ = 5.404119e+00; *vtp++ = 1.277405e+06 * CM_PER_S; *vrp++ = 5.139252e-01 * CM_PER_S; *dp++ = 2.363387e-11 * G_PER_CM3; n++;
    *xp++ = 5.560726e+00; *vtp++ = 1.259215e+06 * CM_PER_S; *vrp++ = 5.145843e-01 * CM_PER_S; *dp++ = 2.211161e-11 * G_PER_CM3; n++;
    *xp++ = 5.721871e+00; *vtp++ = 1.241283e+06 * CM_PER_S; *vrp++ = 5.152500e-01 * CM_PER_S; *dp++ = 2.068716e-11 * G_PER_CM3; n++;
    *xp++ = 5.887687e+00; *vtp++ = 1.223605e+06 * CM_PER_S; *vrp++ = 5.159224e-01 * CM_PER_S; *dp++ = 1.935425e-11 * G_PER_CM3; n++;
    *xp++ = 6.058307e+00; *vtp++ = 1.206177e+06 * CM_PER_S; *vrp++ = 5.166013e-01 * CM_PER_S; *dp++ = 1.810704e-11 * G_PER_CM3; n++;
    *xp++ = 6.233872e+00; *vtp++ = 1.188997e+06 * CM_PER_S; *vrp++ = 5.172868e-01 * CM_PER_S; *dp++ = 1.694001e-11 * G_PER_CM3; n++;
    *xp++ = 6.414525e+00; *vtp++ = 1.172060e+06 * CM_PER_S; *vrp++ = 5.179786e-01 * CM_PER_S; *dp++ = 1.584803e-11 * G_PER_CM3; n++;
    *xp++ = 6.600413e+00; *vtp++ = 1.155363e+06 * CM_PER_S; *vrp++ = 5.186768e-01 * CM_PER_S; *dp++ = 1.482628e-11 * G_PER_CM3; n++;
    *xp++ = 6.791688e+00; *vtp++ = 1.138903e+06 * CM_PER_S; *vrp++ = 5.193813e-01 * CM_PER_S; *dp++ = 1.387027e-11 * G_PER_CM3; n++;
    *xp++ = 6.988506e+00; *vtp++ = 1.122675e+06 * CM_PER_S; *vrp++ = 5.200920e-01 * CM_PER_S; *dp++ = 1.297577e-11 * G_PER_CM3; n++;
    *xp++ = 7.191027e+00; *vtp++ = 1.106678e+06 * CM_PER_S; *vrp++ = 5.208089e-01 * CM_PER_S; *dp++ = 1.213883e-11 * G_PER_CM3; n++;
    *xp++ = 7.399418e+00; *vtp++ = 1.090908e+06 * CM_PER_S; *vrp++ = 5.215319e-01 * CM_PER_S; *dp++ = 1.135576e-11 * G_PER_CM3; n++;
    *xp++ = 7.613847e+00; *vtp++ = 1.075361e+06 * CM_PER_S; *vrp++ = 5.222609e-01 * CM_PER_S; *dp++ = 1.062311e-11 * G_PER_CM3; n++;
    *xp++ = 7.834490e+00; *vtp++ = 1.060035e+06 * CM_PER_S; *vrp++ = 5.229960e-01 * CM_PER_S; *dp++ = 9.937633e-12 * G_PER_CM3; n++;
    *xp++ = 8.061528e+00; *vtp++ = 1.044926e+06 * CM_PER_S; *vrp++ = 5.237370e-01 * CM_PER_S; *dp++ = 9.296296e-12 * G_PER_CM3; n++;
    *xp++ = 8.295145e+00; *vtp++ = 1.030031e+06 * CM_PER_S; *vrp++ = 5.244840e-01 * CM_PER_S; *dp++ = 8.696270e-12 * G_PER_CM3; n++;
    *xp++ = 8.535532e+00; *vtp++ = 1.015346e+06 * CM_PER_S; *vrp++ = 5.252367e-01 * CM_PER_S; *dp++ = 8.134904e-12 * G_PER_CM3; n++;
    *xp++ = 8.782885e+00; *vtp++ = 1.000870e+06 * CM_PER_S; *vrp++ = 5.259953e-01 * CM_PER_S; *dp++ = 7.609701e-12 * G_PER_CM3; n++;
    *xp++ = 9.037406e+00; *vtp++ = 9.865996e+05 * CM_PER_S; *vrp++ = 5.267596e-01 * CM_PER_S; *dp++ = 7.118348e-12 * G_PER_CM3; n++;
    *xp++ = 9.299303e+00; *vtp++ = 9.725309e+05 * CM_PER_S; *vrp++ = 5.275295e-01 * CM_PER_S; *dp++ = 6.658660e-12 * G_PER_CM3; n++;
    *xp++ = 9.568789e+00; *vtp++ = 9.586617e+05 * CM_PER_S; *vrp++ = 5.283052e-01 * CM_PER_S; *dp++ = 6.228604e-12 * G_PER_CM3; n++;
    *xp++ = 9.846085e+00; *vtp++ = 9.449890e+05 * CM_PER_S; *vrp++ = 5.290864e-01 * CM_PER_S; *dp++ = 5.826278e-12 * G_PER_CM3; n++;
    *xp++ = 1.013142e+01; *vtp++ = 9.315100e+05 * CM_PER_S; *vrp++ = 5.298732e-01 * CM_PER_S; *dp++ = 5.449891e-12 * G_PER_CM3; n++;
    *xp++ = 1.042502e+01; *vtp++ = 9.182220e+05 * CM_PER_S; *vrp++ = 5.306655e-01 * CM_PER_S; *dp++ = 5.097776e-12 * G_PER_CM3; n++;
    *xp++ = 1.072713e+01; *vtp++ = 9.051223e+05 * CM_PER_S; *vrp++ = 5.314632e-01 * CM_PER_S; *dp++ = 4.768377e-12 * G_PER_CM3; n++;
    *xp++ = 1.103799e+01; *vtp++ = 8.922082e+05 * CM_PER_S; *vrp++ = 5.322664e-01 * CM_PER_S; *dp++ = 4.460224e-12 * G_PER_CM3; n++;
    *xp++ = 1.135786e+01; *vtp++ = 8.794771e+05 * CM_PER_S; *vrp++ = 5.330749e-01 * CM_PER_S; *dp++ = 4.171953e-12 * G_PER_CM3; n++;
    *xp++ = 1.168700e+01; *vtp++ = 8.669264e+05 * CM_PER_S; *vrp++ = 5.338887e-01 * CM_PER_S; *dp++ = 3.902287e-12 * G_PER_CM3; n++;
    *xp++ = 1.202569e+01; *vtp++ = 8.545535e+05 * CM_PER_S; *vrp++ = 5.347079e-01 * CM_PER_S; *dp++ = 3.650021e-12 * G_PER_CM3; n++;
    *xp++ = 1.237418e+01; *vtp++ = 8.423559e+05 * CM_PER_S; *vrp++ = 5.355322e-01 * CM_PER_S; *dp++ = 3.414037e-12 * G_PER_CM3; n++;
    *xp++ = 1.273278e+01; *vtp++ = 8.303312e+05 * CM_PER_S; *vrp++ = 5.363618e-01 * CM_PER_S; *dp++ = 3.193288e-12 * G_PER_CM3; n++;
    *xp++ = 1.310176e+01; *vtp++ = 8.184768e+05 * CM_PER_S; *vrp++ = 5.371966e-01 * CM_PER_S; *dp++ = 2.986791e-12 * G_PER_CM3; n++;
    *xp++ = 1.348144e+01; *vtp++ = 8.067904e+05 * CM_PER_S; *vrp++ = 5.380365e-01 * CM_PER_S; *dp++ = 2.793627e-12 * G_PER_CM3; n++;
    *xp++ = 1.387212e+01; *vtp++ = 7.952695e+05 * CM_PER_S; *vrp++ = 5.388814e-01 * CM_PER_S; *dp++ = 2.612937e-12 * G_PER_CM3; n++;
    *xp++ = 1.427412e+01; *vtp++ = 7.839118e+05 * CM_PER_S; *vrp++ = 5.397315e-01 * CM_PER_S; *dp++ = 2.443917e-12 * G_PER_CM3; n++;
    *xp++ = 1.468778e+01; *vtp++ = 7.727151e+05 * CM_PER_S; *vrp++ = 5.405865e-01 * CM_PER_S; *dp++ = 2.285815e-12 * G_PER_CM3; n++;
    *xp++ = 1.511342e+01; *vtp++ = 7.616770e+05 * CM_PER_S; *vrp++ = 5.414465e-01 * CM_PER_S; *dp++ = 2.137926e-12 * G_PER_CM3; n++;
    *xp++ = 1.555139e+01; *vtp++ = 7.507952e+05 * CM_PER_S; *vrp++ = 5.423115e-01 * CM_PER_S; *dp++ = 1.999593e-12 * G_PER_CM3; n++;
    *xp++ = 1.600206e+01; *vtp++ = 7.400677e+05 * CM_PER_S; *vrp++ = 5.431813e-01 * CM_PER_S; *dp++ = 1.870198e-12 * G_PER_CM3; n++;
    *xp++ = 1.646579e+01; *vtp++ = 7.294920e+05 * CM_PER_S; *vrp++ = 5.440561e-01 * CM_PER_S; *dp++ = 1.749165e-12 * G_PER_CM3; n++;
    *xp++ = 1.694295e+01; *vtp++ = 7.190662e+05 * CM_PER_S; *vrp++ = 5.449357e-01 * CM_PER_S; *dp++ = 1.635954e-12 * G_PER_CM3; n++;
    *xp++ = 1.743395e+01; *vtp++ = 7.087881e+05 * CM_PER_S; *vrp++ = 5.458201e-01 * CM_PER_S; *dp++ = 1.530062e-12 * G_PER_CM3; n++;
    *xp++ = 1.793917e+01; *vtp++ = 6.986556e+05 * CM_PER_S; *vrp++ = 5.467092e-01 * CM_PER_S; *dp++ = 1.431015e-12 * G_PER_CM3; n++;
    *xp++ = 1.845903e+01; *vtp++ = 6.886666e+05 * CM_PER_S; *vrp++ = 5.476032e-01 * CM_PER_S; *dp++ = 1.338371e-12 * G_PER_CM3; n++;
    STD_PROF_N = n;
    DRAG_COEFF = atof(argv[4]);

    struct reb_simulation *reb_sim = reb_create_simulation();

    reb_sim->integrator = REB_INTEGRATOR_IAS15;
    reb_sim->G = 4. * IOPF_PI * IOPF_PI;
    // reb_sim->collision = REB_COLLISION_DIRECT;
    // reb_sim->collision_resolve = reb_collision_resolve_merge;

    reb_add_fmt(reb_sim, "m", 1.);               // Central object of 1 solar mass

    double m = 3e-6 * atof(argv[1]);
    double rho_p = 3.0 * G_PER_CM3;
    double r = cbrt(m * 3 / (rho_p * 4 * IOPF_PI));

    reb_add_fmt(reb_sim, "m a e r", m, atof(argv[2]), atof(argv[3]), r);

    reb_sim->heartbeat = heartbeat;
    reb_sim->additional_forces = IOPF_drag_all;
    reb_sim->force_is_velocity_dependent = 1;

    reb_move_to_com(reb_sim);

    struct reb_orbit orbit = reb_tools_particle_to_orbit(reb_sim->G, reb_sim->particles[1], reb_sim->particles[0]);
    aI = orbit.a;
    eI = orbit.e;
    EI=reb_tools_energy(reb_sim);

    reb_integrate(reb_sim, 1e5);

    reb_free_simulation(reb_sim);

    if (reb_sim->status == REB_EXIT_SUCCESS) {
        FILE* DONE = fopen("DONE", "w");
        fclose(DONE);
    }
}
