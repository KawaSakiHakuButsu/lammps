// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Bin Jin (Peking university)
   Contact Email: 2201112399@pku.edu.cn
------------------------------------------------------------------------- */

#include "temper_muvt.h"

#include "atom.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "finish.h"
#include "fix.h"
#include "force.h"
#include "integrate.h"
#include "modify.h"
#include "random_park.h"
#include "timer.h"
#include "universe.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

#define TEMPER_DEBUG 0

/* ---------------------------------------------------------------------- */

TemperMuVT::TemperMuVT(LAMMPS *lmp) : Command(lmp), tempfix(nullptr) {}

/* ---------------------------------------------------------------------- */

TemperMuVT::~TemperMuVT()
{
  MPI_Comm_free(&roots);
  if (ranswap) delete ranswap;
  delete ranboltz;
  delete[] set_temp;
  delete[] set_mu;
  delete[] temp2world;
  delete[] world2temp;
  delete[] world2root;
}

/* ----------------------------------------------------------------------
   perform tempering with inter-world swaps
------------------------------------------------------------------------- */

void TemperMuVT::command(int narg, char **arg)
{
  if (universe->nworlds == 1)
    error->universe_all(FLERR,"More than one processor partition required for temper/muvt command");
  if (domain->box_exist == 0)
    error->universe_all(FLERR,"Temper/muvt command before simulation box is defined");
  if (narg != 7 && narg != 8 && narg != 9 && narg != 10) error->universe_all(FLERR,"Illegal temper/muvt command");

  int nsteps = utils::inumeric(FLERR,arg[0],false,lmp);
  nevery = utils::inumeric(FLERR,arg[1],false,lmp);
  double temp = utils::numeric(FLERR,arg[2],false,lmp);
  double mu = utils::numeric(FLERR,arg[3],false,lmp);

  // ignore temper command, if walltime limit was already reached

  if (timer->is_timeout()) return;
  
  gcmcfix = modify->get_fix_by_id(arg[4]);
  if (!gcmcfix)
    error->universe_all(FLERR,fmt::format("Tempering fix ID {} is not defined", arg[4]));

  seed_swap = utils::inumeric(FLERR,arg[5],false,lmp);
  seed_boltz = utils::inumeric(FLERR,arg[6],false,lmp);

  my_set_temp_mu = universe->iworld;
  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"temp-fix") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal temper/muvt command");
      tempfix = modify->get_fix_by_id(arg[iarg+1]);
      if (!tempfix)
        error->universe_all(FLERR,fmt::format("Tempering fix ID {} is not defined", arg[iarg+1]));
      iarg += 2;
    }
    else {
      my_set_temp_mu = utils::inumeric(FLERR,arg[iarg],false,lmp);
      iarg++;
    }
  }
  if ((my_set_temp_mu < 0) || (my_set_temp_mu >= universe->nworlds))
    error->universe_one(FLERR,"Invalid temperature and mu index value");

  // swap frequency must evenly divide total # of timesteps

  if (nevery <= 0)
    error->universe_all(FLERR,"Invalid frequency in temper/muvt command");
  nswaps = nsteps/nevery;
  if (nswaps*nevery != nsteps)
    error->universe_all(FLERR,"Non integer # of swaps in temper/muvt command");

  // fix style must be appropriate for temperature control and gcmc, i.e. it needs
  // to provide a working Fix::reset_target() and must not change the volume.

  if (!utils::strmatch(gcmcfix->style,"^gcmc")) error->universe_all(FLERR, "Grand canonical Monte Carlo fix is not supported");

  if (tempfix)
    if ((!utils::strmatch(tempfix->style,"^nvt")) &&
        (!utils::strmatch(tempfix->style,"^langevin")) &&
        (!utils::strmatch(tempfix->style,"^gl[de]$")) &&
        (!utils::strmatch(tempfix->style,"^rigid/nvt")) &&
        (!utils::strmatch(tempfix->style,"^temp/")))
      error->universe_all(FLERR,"Tempering temperature fix is not supported");

  // setup for long tempering run

  update->whichflag = 1;
  timer->init_timeout();

  update->nsteps = nsteps;
  update->beginstep = update->firststep = update->ntimestep;
  update->endstep = update->laststep = update->firststep + nsteps;
  if (update->laststep < 0) error->all(FLERR,"Too many timesteps");

  lmp->init();

  // local storage

  me_universe = universe->me;
  MPI_Comm_rank(world,&me);
  nworlds = universe->nworlds;
  iworld = universe->iworld;
  boltz = force->boltz;

  // pe_compute = ptr to thermo_pe compute
  // notify compute it will be called at first swap

  Compute *pe_compute = modify->get_compute_by_id("thermo_pe");
  if (!pe_compute) error->all(FLERR,"Tempering could not find thermo_pe compute");

  pe_compute->addstep(update->ntimestep + nevery);

  // create MPI communicator for root proc from each world

  int color;
  if (me == 0) color = 0;
  else color = 1;
  MPI_Comm_split(universe->uworld,color,0,&roots);

  // RNGs for swaps and Boltzmann test
  // warm up Boltzmann RNG

  if (seed_swap) ranswap = new RanPark(lmp,seed_swap);
  else ranswap = nullptr;
  ranboltz = new RanPark(lmp,seed_boltz + me_universe);
  for (int i = 0; i < 100; i++) ranboltz->uniform();

  // world2root[i] = global proc that is root proc of world i

  world2root = new int[nworlds];
  if (me == 0)
    MPI_Allgather(&me_universe,1,MPI_INT,world2root,1,MPI_INT,roots);
  MPI_Bcast(world2root,nworlds,MPI_INT,0,world);

  // create static list of set temperatures
  // allgather tempering arg "temp" across root procs
  // bcast from each root to other procs in world

  set_temp = new double[nworlds];
  if (me == 0) MPI_Allgather(&temp,1,MPI_DOUBLE,set_temp,1,MPI_DOUBLE,roots);
  MPI_Bcast(set_temp,nworlds,MPI_DOUBLE,0,world);

  // create static list of set chemical potentials
  // allgather tempering arg "mu" across root procs
  // bcast from each root to other procs in world

  set_mu = new double[nworlds];
  if (me == 0) MPI_Allgather(&mu,1,MPI_DOUBLE,set_mu,1,MPI_DOUBLE,roots);
  MPI_Bcast(set_mu,nworlds,MPI_DOUBLE,0,world);

  // create world2temp only on root procs from my_set_temp_mu
  // create temp2world on root procs from world2temp,
  //   then bcast to all procs within world

  world2temp = new int[nworlds];
  temp2world = new int[nworlds];
  if (me == 0) {
    MPI_Allgather(&my_set_temp_mu,1,MPI_INT,world2temp,1,MPI_INT,roots);
    for (int i = 0; i < nworlds; i++) temp2world[world2temp[i]] = i;
  }
  MPI_Bcast(temp2world,nworlds,MPI_INT,0,world);

  // if restarting tempering, reset temp target of Fix to current my_set_temp_mu

  if (narg == 8 || narg == 10) {
    double new_temp = set_temp[my_set_temp_mu];
    double new_mu = set_mu[my_set_temp_mu];
    gcmcfix->reset_target(new_temp);
    gcmcfix->reset_mu(new_mu);
    if (tempfix) tempfix->reset_target(new_temp);
  }

  // setup tempering runs

  int i,which,partner,swap,partner_set_temp_mu,partner_world;
  double pe,pe_partner,boltz_factor,new_temp,new_mu;
  int natom, natom_partner;

  if (me_universe == 0 && universe->uscreen)
    fprintf(universe->uscreen,"Setting up tempering ...\n");

  update->integrate->setup(1);

  if (me_universe == 0) {
    if (universe->uscreen) {
      fprintf(universe->uscreen,"Step");
      for (int i = 0; i < nworlds; i++)
        fprintf(universe->uscreen," T%d",i);
      fprintf(universe->uscreen,"\n");
    }
    if (universe->ulogfile) {
      fprintf(universe->ulogfile,"Step");
      for (int i = 0; i < nworlds; i++)
        fprintf(universe->ulogfile," T%d",i);
      fprintf(universe->ulogfile,"\n");
    }
    print_status();
  }

  timer->init();
  timer->barrier_start();

  for (int iswap = 0; iswap < nswaps; iswap++) {

    // run for nevery timesteps

    timer->init_timeout();
    update->integrate->run(nevery);

    // check for timeout across all procs

    int my_timeout=0;
    int any_timeout=0;
    if (timer->is_timeout()) my_timeout=1;
    MPI_Allreduce(&my_timeout, &any_timeout, 1, MPI_INT, MPI_SUM, universe->uworld);
    if (any_timeout) {
      timer->force_timeout();
      break;
    }

    // compute PE
    // notify compute it will be called at next swap

    pe = pe_compute->compute_scalar();
    pe_compute->addstep(update->ntimestep + nevery);

    // obtain # of atoms in the system

    natom = atom->natoms;

    // which = which of 2 kinds of swaps to do (0,1)

    if (!ranswap) which = iswap % 2;
    else if (ranswap->uniform() < 0.5) which = 0;
    else which = 1;

    // partner_set_temp_mu = which set temp and mu I am partnering with for this swap

    if (which == 0) {
      if (my_set_temp_mu % 2 == 0) partner_set_temp_mu = my_set_temp_mu + 1;
      else partner_set_temp_mu = my_set_temp_mu - 1;
    } else {
      if (my_set_temp_mu % 2 == 1) partner_set_temp_mu = my_set_temp_mu + 1;
      else partner_set_temp_mu = my_set_temp_mu - 1;
    }

    // partner = proc ID to swap with
    // if partner = -1, then I am not a proc that swaps

    partner = -1;
    if (me == 0 && partner_set_temp_mu >= 0 && partner_set_temp_mu < nworlds) {
      partner_world = temp2world[partner_set_temp_mu];
      partner = world2root[partner_world];
    }

    // swap with a partner, only root procs in each world participate
    // hi proc sends PE to low proc
    // lo proc make Boltzmann decision on whether to swap
    // lo proc communicates decision back to hi proc

    swap = 0;
    if (partner != -1) {
      if (me_universe > partner)
        MPI_Send(&pe,1,MPI_DOUBLE,partner,0,universe->uworld);
      else
        MPI_Recv(&pe_partner,1,MPI_DOUBLE,partner,0,universe->uworld,MPI_STATUS_IGNORE);

      if (me_universe > partner)
        MPI_Send(&natom,1,MPI_INT,partner,0,universe->uworld);
      else
        MPI_Recv(&natom_partner,1,MPI_INT,partner,0,universe->uworld,MPI_STATUS_IGNORE);

    // Acceptance criteria changed versus temper command for MuVT ensemble
      if (me_universe < partner) {
        boltz_factor = (pe - pe_partner) *
          (1.0/(boltz*set_temp[my_set_temp_mu]) -
           1.0/(boltz*set_temp[partner_set_temp_mu])) -
          (natom - natom_partner) *
          (set_mu[my_set_temp_mu]/(boltz*set_temp[my_set_temp_mu]) -
           set_mu[partner_set_temp_mu]/(boltz*set_temp[partner_set_temp_mu]));
        if (boltz_factor >= 0.0) swap = 1;
        else if (ranboltz->uniform() < exp(boltz_factor)) swap = 1;
      }

      if (me_universe < partner)
        MPI_Send(&swap,1,MPI_INT,partner,0,universe->uworld);
      else
        MPI_Recv(&swap,1,MPI_INT,partner,0,universe->uworld,MPI_STATUS_IGNORE);

#if TEMPER_DEBUG
      if (me_universe < partner)
        fprintf(universe->uscreen,"SWAP %d & %d: yes = %d,Ts = %d %d, PEs = %g %g, Bz = %g %g, atoms = %d %d\n",
                me_universe,partner,swap,my_set_temp_mu,partner_set_temp_mu,
                pe,pe_partner,boltz_factor,exp(boltz_factor), natom, natom_partner);
#endif
    }

    // bcast swap result to other procs in my world

    MPI_Bcast(&swap,1,MPI_INT,0,world);

    // rescale kinetic energy via velocities if move is accepted

    if (swap) scale_velocities(partner_set_temp_mu,my_set_temp_mu);

    // if my world swapped, all procs in world reset temp target of Fix

    if (swap) {
      new_temp = set_temp[partner_set_temp_mu];
      new_mu = set_mu[partner_set_temp_mu];
      gcmcfix->reset_target(new_temp);
      gcmcfix->reset_mu(new_mu);
      if (tempfix) tempfix->reset_target(new_temp);
    }

    // update my_set_temp_mu and temp2world on every proc
    // root procs update their value if swap took place
    // allgather across root procs
    // bcast within my world

    if (swap) my_set_temp_mu = partner_set_temp_mu;
    if (me == 0) {
      MPI_Allgather(&my_set_temp_mu,1,MPI_INT,world2temp,1,MPI_INT,roots);
      for (i = 0; i < nworlds; i++) temp2world[world2temp[i]] = i;
    }
    MPI_Bcast(temp2world,nworlds,MPI_INT,0,world);

    // print out current swap status

    if (me_universe == 0) print_status();
  }

  timer->barrier_stop();

  update->integrate->cleanup();

  Finish finish(lmp);
  finish.end(1);

  update->whichflag = 0;
  update->firststep = update->laststep = 0;
  update->beginstep = update->endstep = 0;
}

/* ----------------------------------------------------------------------
   scale kinetic energy via velocities a la Sugita
------------------------------------------------------------------------- */

void TemperMuVT::scale_velocities(int t_partner, int t_me)
{
  double sfactor = sqrt(set_temp[t_partner]/set_temp[t_me]);

  double **v = atom->v;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    v[i][0] = v[i][0]*sfactor;
    v[i][1] = v[i][1]*sfactor;
    v[i][2] = v[i][2]*sfactor;
  }
}

/* ----------------------------------------------------------------------
   proc 0 prints current tempering status
------------------------------------------------------------------------- */

void TemperMuVT::print_status()
{
  std::string status = std::to_string(update->ntimestep);
  for (int i = 0; i < nworlds; i++)
    status += " " + std::to_string(world2temp[i]);

  status += "\n";

  if (universe->uscreen) fputs(status.c_str(), universe->uscreen);
  if (universe->ulogfile) {
    fputs(status.c_str(), universe->ulogfile);
    fflush(universe->ulogfile);
  }
}
