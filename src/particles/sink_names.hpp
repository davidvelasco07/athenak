#ifndef PARTICLES_SINK_NAMES_HPP_
#define PARTICLES_SINK_NAMES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink_names.hpp
//! \brief unqualified sink field names, for the sink implementation and sink problem
//! generators only. IPX/IPY/IPZ and PGID/PTAG/PSTATUS are the common names already
//! declared in particles.hpp, with the same values.

#include "particles/sink.hpp"

using particles::sink::IPVX;
using particles::sink::IPVY;
using particles::sink::IPVZ;
using particles::sink::IPM;
using particles::sink::IPGX;
using particles::sink::IPGY;
using particles::sink::IPGZ;
using particles::sink::IPX0;
using particles::sink::IPY0;
using particles::sink::IPZ0;
using particles::sink::IPVX0;
using particles::sink::IPVY0;
using particles::sink::IPVZ0;

#endif // PARTICLES_SINK_NAMES_HPP_
