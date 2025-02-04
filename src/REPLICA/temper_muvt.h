/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMMAND_CLASS
// clang-format off
CommandStyle(temper/muvt,TemperMuVT);
// clang-format on
#else

#ifndef LMP_TEMPERMUVT_H
#define LMP_TEMPERMUVT_H

#include "command.h"

namespace LAMMPS_NS {

class TemperMuVT : public Command {
 public:
  TemperMuVT(class LAMMPS *);
  ~TemperMuVT() override;
  void command(int, char **) override;

 private:
  int me, me_universe;                  // my proc ID in world and universe
  int iworld, nworlds;                  // world info
  double boltz;                         // copy from output->boltz
  MPI_Comm roots;                       // MPI comm with 1 root proc from each world
  class RanPark *ranswap, *ranboltz;    // RNGs for swapping and Boltz factor
  int nevery;                           // # of timesteps between swaps
  int nswaps;                           // # of tempering swaps to perform
  int seed_swap;                        // 0 = toggle swaps, n = RNG for swap direction
  int seed_boltz;                       // seed for Boltz factor comparison
  class Fix *gcmcfix;                   // grand canonical Monte Carlo fix to use
  class Fix *tempfix;                  // temperature fix to use

  int my_set_temp_mu;     // which set temp and mu I am simulating
  double *set_temp;    // static list of replica set temperatures
  double *set_mu;      // static list of replica set chemical potentials
  int *temp2world;     // temp2world[i] = world simulating set temp i
  int *world2temp;     // world2temp[i] = temp simulated by world i
  int *world2root;     // world2root[i] = root proc of world i

  void scale_velocities(int, int);
  void print_status();
};

}    // namespace LAMMPS_NS

#endif
#endif
