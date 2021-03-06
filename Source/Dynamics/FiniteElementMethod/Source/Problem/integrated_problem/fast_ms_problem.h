#ifndef PhysIKA_FAST_MS_PROBLEM
#define PhysIKA_FAST_MS_PROBLEM
#include <boost/property_tree/ptree.hpp>

#include "Common/framework.h"
#include "Problem/constraint/constraints.h"
#include "Problem/energy/basic_energy.h"
#include "Geometry/embedded_interpolate.h"
#include "mass_spring_problem.h"
#include "embedded_mass_spring_problem.h"
#include "Solver/fast_ms_solver.h"

namespace PhysIKA{

template<typename T>
class fast_ms_builder : public embedded_ms_problem_builder<T>
{
 public:
  fast_ms_builder(const T*x, const boost::property_tree::ptree& pt); 

  std::shared_ptr<fast_ms_info<T>> get_fast_ms_solver_info() const { return solver_info_;}
  virtual int update_problem(const T* x, const T* v = nullptr);
  
  using embedded_ms_problem_builder<T>::REST_;
  using embedded_ms_problem_builder<T>::cells_;
  using embedded_ms_problem_builder<T>::collider_;
  using embedded_ms_problem_builder<T>::kinetic_;
  using embedded_ms_problem_builder<T>::ebf_;
  using embedded_ms_problem_builder<T>::cbf_;
  using embedded_ms_problem_builder<T>::pt_;
  using embedded_ms_problem_builder<T>::get_collider;
  using embedded_ms_problem_builder<T>::get_cells;
  using embedded_ms_problem_builder<T>::get_nods;
  using embedded_ms_problem_builder<T>::build_problem;
  using embedded_ms_problem_builder<T>::semi_implicit_;


  using embedded_ms_problem_builder<T>::fine_verts_num_;
  using embedded_ms_problem_builder<T>::embedded_interp_;
  using embedded_ms_problem_builder<T>::coarse_to_fine_coef_; // coarse * coef = fine
  using embedded_ms_problem_builder<T>::fine_to_coarse_coef_; // fine * coef = coarse

protected:
  std::shared_ptr<fast_ms_info<T>> solver_info_;
};
}
#endif
