#pragma once
#include "Core/DataTypes.h"
#include "Core/Utility.h"
#include "Core/Array/Array.h"
#include "Framework/Topology/NeighborList.h"
//#include "Framework/Topology/Primitive3D.h"
#include "Core/Utility/Scan.h"
#include "EdgeSet.h"
#include "Framework/Framework/ModuleTopology.h"
#include "Framework/Topology/Primitive3D.h"

namespace PhysIKA {

#define INVALID -1
#define BUCKETS 8
#define CAPACITY 16

	template<typename TDataType>
	class GridHash
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		GridHash();
		~GridHash();


		void setSpace(Real _h, Coord _lo, Coord _hi);

		void construct(DeviceArray<Coord>& pos);
		void construct(DeviceArray<Coord>& pos, DeviceArray<Triangle>& tri, DeviceArray<Coord>& Tri_pos);

		void initializeMultiLevel();

		void clear();

		void release();

		GPU_FUNC inline int getIndex(int i, int j, int k)
		{
			if (i < 0 || i >= nx) return INVALID;
			if (j < 0 || j >= ny) return INVALID;
			if (k < 0 || k >= nz) return INVALID;

			return i + j * nx + k * nx * ny;
		}

		GPU_FUNC inline int getIndex(Coord pos)
		{
			int i = floor((pos[0] - lo[0]) / ds);
			int j = floor((pos[1] - lo[1]) / ds);
			int k = floor((pos[2] - lo[2]) / ds);

			return getIndex(i, j, k);
		}

		GPU_FUNC inline int getIndex(int i, int j, int k, int level)
		{
			if (i < 0 || i >= nx_list[level]) return INVALID;
			if (j < 0 || j >= ny_list[level]) return INVALID;
			if (k < 0 || k >= nz_list[level]) return INVALID;
			if (level >= 20) return INVALID;

			if (level >= 1)
				return i + j * nx_list[level] + k * nx_list[level] * ny_list[level] + prefix[level - 1];
			else
				return i + j * nx_list[level] + k * nx_list[level] * ny_list[level];
		}


		GPU_FUNC inline int getIndex(Coord pos, int level)
		{
			int i = floor((pos[0] - lo[0]) / (ds * (Real)(1 << level)));
			int j = floor((pos[1] - lo[1]) / (ds * (Real)(1 << level)));
			int k = floor((pos[2] - lo[2]) / (ds * (Real)(1 << level)));

			return getIndex(i, j, k, level);
		}

		GPU_FUNC inline int3 getIndex3(Coord pos)
		{
			int i = floor((pos[0] - lo[0]) / ds);
			int j = floor((pos[1] - lo[1]) / ds);
			int k = floor((pos[2] - lo[2]) / ds);

			return make_int3(i, j, k);
		}

		GPU_FUNC inline int getCounter(int gId) {
			if (gId >= num - 1)
			{
				return particle_num - index[gId];
			}
			return index[gId + 1] - index[gId];
			//return counter[gId]; 
		}

		GPU_FUNC inline int getParticleId(int gId, int n) {
			return ids[index[gId] + n];
		}

	public:
		int num;
		int nx, ny, nz;
		bool multi_grid = false;

		int particle_num = 0;
		int maxlevel = 0;

		int prefix[20];
		int nx_list[20], ny_list[20], nz_list[20];

		Real ds;

		Coord lo;
		Coord hi;

		//int npMax;		//maximum particle number for each cell

		int* ids = nullptr;
		int* counter = nullptr;
		int* index = nullptr;

		Scan* m_scan = nullptr;
		Reduction<int>* m_reduce = nullptr;
	};

#ifdef PRECISION_FLOAT
	template class GridHash<DataType3f>;
#else
	template class GridHash<DataType3d>;
#endif
}