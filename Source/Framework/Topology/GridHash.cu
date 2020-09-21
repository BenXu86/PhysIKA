#pragma once
#include "GridHash.h"
#include "Core/Utility.h"
#include "Primitive3D.h"

namespace PhysIKA {

	__constant__ int offset[27][3] = { 0, 0, 0,
		0, 0, 1,
		0, 1, 0,
		1, 0, 0,
		0, 0, -1,
		0, -1, 0,
		-1, 0, 0,
		0, 1, 1,
		0, 1, -1,
		0, -1, 1,
		0, -1, -1,
		1, 0, 1,
		1, 0, -1,
		-1, 0, 1,
		-1, 0, -1,
		1, 1, 0,
		1, -1, 0,
		-1, 1, 0,
		-1, -1, 0,
		1, 1, 1,
		1, 1, -1,
		1, -1, 1,
		-1, 1, 1,
		1, -1, -1,
		-1, 1, -1,
		-1, -1, 1,
		-1, -1, -1
	};

	template<typename TDataType>
	GridHash<TDataType>::GridHash()
	{
	}

	template<typename TDataType>
	GridHash<TDataType>::~GridHash()
	{
	}

	template<typename TDataType>
	void GridHash<TDataType>::setSpace(Real _h, Coord _lo, Coord _hi)
	{
		release();

		int padding = 2;
		ds = _h;
		lo = _lo - padding * ds;

		Coord nSeg = (_hi - _lo) / ds;

		nx = ceil(nSeg[0]) + 1 + 2 * padding;
		ny = ceil(nSeg[1]) + 1 + 2 * padding;
		nz = ceil(nSeg[2]) + 1 + 2 * padding;
		hi = lo + Coord(nx, ny, nz) * ds;

		num = nx * ny * nz;

		//		npMax = 128;

		cuSafeCall(cudaMalloc((void**)&counter, num * sizeof(int)));
		cuSafeCall(cudaMalloc((void**)&index, num * sizeof(int)));

		if (m_reduce != nullptr)
		{
			delete m_reduce;
		}

		m_reduce = Reduction<int>::Create(num);
		if (multi_grid)
			initializeMultiLevel();
	}
	template<typename TDataType>
	void GridHash<TDataType>::initializeMultiLevel()
	{
		release();
		int level = 0;
		Coord tmp = hi - lo;
		Real maxx = max(tmp[0], tmp[1]);
		maxx = max(maxx, tmp[2]);
		int padding = 2;

		while ((1 << level) * ds < maxx && level < 10)
		{
			Real ds_i = (1 << level) * ds;
			Coord nSeg = (hi - lo) / ds_i;

			nx = ceil(nSeg[0]) + 1 + 2 * padding;
			ny = ceil(nSeg[1]) + 1 + 2 * padding;
			nz = ceil(nSeg[2]) + 1 + 2 * padding;

			int num_i = nx * ny * nz;
			nx_list[level] = nx;
			ny_list[level] = ny;
			nz_list[level] = nz;

			prefix[level] = num_i;
			if (level >= 1)
				prefix[level] += prefix[level - 1];
			level++;
		};
		level--;
		num = prefix[level];
		maxlevel = level;

		cuSafeCall(cudaMalloc((void**)&counter, num * sizeof(int)));
		cuSafeCall(cudaMalloc((void**)&index, num * sizeof(int)));
		if (m_reduce != nullptr)
		{
			delete m_reduce;
		}

		m_reduce = Reduction<int>::Create(num);

	}

	template<typename TDataType>
	__global__ void K_CalculateParticleNumber(GridHash<TDataType> hash, Array<typename TDataType::Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		int gId = hash.getIndex(pos[pId]);

		if (gId != INVALID)
			atomicAdd(&(hash.index[gId]), 1);
	}


	template<typename TDataType>
	__global__ void K_AddTriNumber(GridHash<TDataType> hash, Array<typename TopologyModule::Triangle> tri, Array<typename TDataType::Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= tri.size()) return;
		//printf("TRISIZE:%.3lf %.3lf %.3lf\n",hash.lo[0], hash.lo[1], hash.lo[2]);
		/*
		int gId1 = hash.getIndex(pos[tri[pId][0]]);
		int gId2 = hash.getIndex(pos[tri[pId][1]]);
		int gId3 = hash.getIndex(pos[tri[pId][2]]);
		*/
		Real ds = hash.ds;
		//Coord3D lo = hash.lo;
		//Coord3D hi = hash.hi;

		int i0 = floor((pos[tri[pId][0]][0] - hash.lo[0]) / hash.ds);
		int j0 = floor((pos[tri[pId][0]][1] - hash.lo[1]) / hash.ds);
		int k0 = floor((pos[tri[pId][0]][2] - hash.lo[2]) / hash.ds);

		int i1 = floor((pos[tri[pId][1]][0] - hash.lo[0]) / hash.ds);
		int j1 = floor((pos[tri[pId][1]][1] - hash.lo[1]) / hash.ds);
		int k1 = floor((pos[tri[pId][1]][2] - hash.lo[2]) / hash.ds);

		int i2 = floor((pos[tri[pId][2]][0] - hash.lo[0]) / hash.ds);
		int j2 = floor((pos[tri[pId][2]][1] - hash.lo[1]) / hash.ds);
		int k2 = floor((pos[tri[pId][2]][2] - hash.lo[2]) / hash.ds);

		int imin = i0 < i1 ? i0 : i1;
		imin = i2 < imin ? i2 : imin;
		int imax = i0 > i1 ? i0 : i1;
		imax = i2 > imax ? i2 : imax;

		int jmin = j0 < j1 ? j0 : j1;
		jmin = j2 < jmin ? j2 : jmin;
		int jmax = j0 > j1 ? j0 : j1;
		jmax = j2 > jmax ? j2 : jmax;

		int kmin = k0 < k1 ? k0 : k1;
		kmin = k2 < kmin ? k2 : kmin;
		int kmax = k0 > k1 ? k0 : k1;
		kmax = k2 > kmax ? k2 : kmax;

		imin--; jmin--; kmin--;
		imax++; jmax++; kmax++;

		int addi, addj, addk;
		addi = int(sqrt((Real)imax - (Real)imin + 1));
		addj = int(sqrt((Real)jmax - (Real)jmin + 1));
		addk = int(sqrt((Real)kmax - (Real)kmin + 1));

		Triangle3D t3d = Triangle3D(pos[tri[pId][0]], pos[tri[pId][1]], pos[tri[pId][2]]);
		//printf("%d %d %d\n",addi,addj,addk);
		for (int li = imin; li <= imax; li += 1)
			for (int lj = jmin; lj <= jmax; lj += 1)
				for (int lk = kmin; lk <= kmax; lk += 1)
				{

					int i = li, j = lj, k = lk;
					Coord3D ABP11 = Coord3D(i * ds + hash.lo[0] - 0.1 * ds * 10.0,
						j * ds + hash.lo[1] - 0.1 * ds * 10.0,
						k * ds + hash.lo[2] - 0.1 * ds * 10.0);
					Coord3D ABP22 = Coord3D(i * ds + ds + hash.lo[0] + 0.1 * ds * 10.0,
						j * ds + ds + hash.lo[1] + 0.1 * ds * 10.0,
						k * ds + ds + hash.lo[2] + 0.1 * ds * 10.0);
					AlignedBox3D AABB2 = AlignedBox3D(ABP11, ABP22);

					if (AABB2.meshInsert(t3d))
					{
						int gId = hash.getIndex(i, j, k);
						if (gId != INVALID)
							atomicAdd(&(hash.index[gId]), 1);

					}

				}

	}

	template<typename TDataType>
	__global__ void K_Multi_AddTriNumber(GridHash<TDataType> hash, Array<typename TopologyModule::Triangle> tri, Array<typename TDataType::Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= tri.size()) return;

		Real ds = hash.ds;
		int level = -1;
		int i0, j0, k0;
		int i1, j1, k1;
		int i2, j2, k2;
		int imin, imax, jmin, jmax, kmin, kmax;
		do {

			level++;
			ds = hash.ds * (Real)(1 << level);

			i0 = floor((pos[tri[pId][0]][0] - hash.lo[0]) / ds);
			j0 = floor((pos[tri[pId][0]][1] - hash.lo[1]) / ds);
			k0 = floor((pos[tri[pId][0]][2] - hash.lo[2]) / ds);

			i1 = floor((pos[tri[pId][1]][0] - hash.lo[0]) / ds);
			j1 = floor((pos[tri[pId][1]][1] - hash.lo[1]) / ds);
			k1 = floor((pos[tri[pId][1]][2] - hash.lo[2]) / ds);

			i2 = floor((pos[tri[pId][2]][0] - hash.lo[0]) / ds);
			j2 = floor((pos[tri[pId][2]][1] - hash.lo[1]) / ds);
			k2 = floor((pos[tri[pId][2]][2] - hash.lo[2]) / ds);

			imin = i0 < i1 ? i0 : i1;
			imin = i2 < imin ? i2 : imin;
			imax = i0 > i1 ? i0 : i1;
			imax = i2 > imax ? i2 : imax;

			jmin = j0 < j1 ? j0 : j1;
			jmin = j2 < jmin ? j2 : jmin;
			jmax = j0 > j1 ? j0 : j1;
			jmax = j2 > jmax ? j2 : jmax;

			kmin = k0 < k1 ? k0 : k1;
			kmin = k2 < kmin ? k2 : kmin;
			kmax = k0 > k1 ? k0 : k1;
			kmax = k2 > kmax ? k2 : kmax;

			imin--; jmin--; kmin--;
			imax++; jmax++; kmax++;

		} while ((imax - imin + 1) * (jmax - jmin + 1) * (kmax - kmin + 1) > 150 && level < hash.maxlevel);


		//ds = hash.ds;
		Triangle3D t3d = Triangle3D(pos[tri[pId][0]], pos[tri[pId][1]], pos[tri[pId][2]]);

		for (int li = imin; li <= imax; li += 1)
			for (int lj = jmin; lj <= jmax; lj += 1)
				for (int lk = kmin; lk <= kmax; lk += 1)
				{
					int i = li, j = lj, k = lk;
					Coord3D ABP11 = Coord3D(i * ds + hash.lo[0] - 0.1 * hash.ds * 10.0,
						j * ds + hash.lo[1] - 0.1 * hash.ds * 10.0,
						k * ds + hash.lo[2] - 0.1 * hash.ds * 10.0);
					Coord3D ABP22 = Coord3D(i * ds + ds + hash.lo[0] + 0.1 * hash.ds * 10.0,
						j * ds + ds + hash.lo[1] + 0.1 * hash.ds * 10.0,
						k * ds + ds + hash.lo[2] + 0.1 * hash.ds * 10.0);
					AlignedBox3D AABB2 = AlignedBox3D(ABP11, ABP22);

					if (AABB2.meshInsert(t3d))
					{
						int gId = hash.getIndex(i, j, k, level);
						//printf("^^^^^^^^^^^^^^^^^^^^^^^^^^^ %d %d %d\n", gId, level, INVALID);
						if (gId != INVALID)
							atomicAdd(&(hash.index[gId]), 1);

					}


				}

	}

	template<typename TDataType>
	__global__ void K_ConstructHashTable(GridHash<TDataType> hash, Array<typename TDataType::Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		int gId = hash.getIndex(pos[pId]);

		if (gId < 0) return;

		int index = atomicAdd(&(hash.counter[gId]), 1);
		// 		index = index < hash.npMax - 1 ? index : hash.npMax - 1;
		// 		hash.ids[gId * hash.npMax + index] = pId;
		hash.ids[hash.index[gId] + index] = pId;
	}


	template<typename TDataType>
	__global__ void K_AddTriElement(GridHash<TDataType> hash, Array<typename TopologyModule::Triangle> tri, Array<typename TDataType::Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= tri.size()) return;
		Real ds = hash.ds;
		//Coord3D lo = hash.lo;
		//Coord3D hi = hash.hi;
		int i0 = floor((pos[tri[pId][0]][0] - hash.lo[0]) / hash.ds);
		int j0 = floor((pos[tri[pId][0]][1] - hash.lo[1]) / hash.ds);
		int k0 = floor((pos[tri[pId][0]][2] - hash.lo[2]) / hash.ds);

		int i1 = floor((pos[tri[pId][1]][0] - hash.lo[0]) / hash.ds);
		int j1 = floor((pos[tri[pId][1]][1] - hash.lo[1]) / hash.ds);
		int k1 = floor((pos[tri[pId][1]][2] - hash.lo[2]) / hash.ds);

		int i2 = floor((pos[tri[pId][2]][0] - hash.lo[0]) / hash.ds);
		int j2 = floor((pos[tri[pId][2]][1] - hash.lo[1]) / hash.ds);
		int k2 = floor((pos[tri[pId][2]][2] - hash.lo[2]) / hash.ds);

		int imin = i0 < i1 ? i0 : i1;
		imin = i2 < imin ? i2 : imin;
		int imax = i0 > i1 ? i0 : i1;
		imax = i2 > imax ? i2 : imax;

		int jmin = j0 < j1 ? j0 : j1;
		jmin = j2 < jmin ? j2 : jmin;
		int jmax = j0 > j1 ? j0 : j1;
		jmax = j2 > jmax ? j2 : jmax;

		int kmin = k0 < k1 ? k0 : k1;
		kmin = k2 < kmin ? k2 : kmin;
		int kmax = k0 > k1 ? k0 : k1;
		kmax = k2 > kmax ? k2 : kmax;
		imin--; jmin--; kmin--;
		imax++; jmax++; kmax++;
		int addi, addj, addk;
		addi = int(sqrt((Real)imax - (Real)imin + 1));
		addj = int(sqrt((Real)jmax - (Real)jmin + 1));
		addk = int(sqrt((Real)kmax - (Real)kmin + 1));

		Triangle3D t3d = Triangle3D(pos[tri[pId][0]], pos[tri[pId][1]], pos[tri[pId][2]]);


		for (int li = imin; li <= imax; li += 1)
			for (int lj = jmin; lj <= jmax; lj += 1)
				for (int lk = kmin; lk <= kmax; lk += 1)
				{

					int i = li, j = lj, k = lk;
					Coord3D ABP11 = Coord3D(i * ds + hash.lo[0] - 0.1 * ds * 10.0,
						j * ds + hash.lo[1] - 0.1 * ds * 10.0,
						k * ds + hash.lo[2] - 0.1 * ds * 10.0);
					Coord3D ABP22 = Coord3D(i * ds + ds + hash.lo[0] + 0.1 * ds * 10.0,
						j * ds + ds + hash.lo[1] + 0.1 * ds * 10.0,
						k * ds + ds + hash.lo[2] + 0.1 * ds * 10.0);
					AlignedBox3D AABB2 = AlignedBox3D(ABP11, ABP22);

					if (AABB2.meshInsert(t3d))
					{
						int gId = hash.getIndex(i, j, k);

						if (gId != INVALID)
						{
							int index = atomicAdd(&(hash.counter[gId]), 1);
							hash.ids[hash.index[gId] + index] = -pId - 1;
						}

					}
				}


	}

	template<typename TDataType>
	__global__ void K_Multi_AddTriElement(GridHash<TDataType> hash, Array<typename TopologyModule::Triangle> tri, Array<typename TDataType::Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= tri.size()) return;

		Real ds = hash.ds;
		int level = -1;
		int i0, j0, k0;
		int i1, j1, k1;
		int i2, j2, k2;
		int imin, imax, jmin, jmax, kmin, kmax;
		do {

			level++;
			ds = hash.ds * (Real)(1 << level);

			i0 = floor((pos[tri[pId][0]][0] - hash.lo[0]) / ds);
			j0 = floor((pos[tri[pId][0]][1] - hash.lo[1]) / ds);
			k0 = floor((pos[tri[pId][0]][2] - hash.lo[2]) / ds);

			i1 = floor((pos[tri[pId][1]][0] - hash.lo[0]) / ds);
			j1 = floor((pos[tri[pId][1]][1] - hash.lo[1]) / ds);
			k1 = floor((pos[tri[pId][1]][2] - hash.lo[2]) / ds);

			i2 = floor((pos[tri[pId][2]][0] - hash.lo[0]) / ds);
			j2 = floor((pos[tri[pId][2]][1] - hash.lo[1]) / ds);
			k2 = floor((pos[tri[pId][2]][2] - hash.lo[2]) / ds);

			imin = i0 < i1 ? i0 : i1;
			imin = i2 < imin ? i2 : imin;
			imax = i0 > i1 ? i0 : i1;
			imax = i2 > imax ? i2 : imax;

			jmin = j0 < j1 ? j0 : j1;
			jmin = j2 < jmin ? j2 : jmin;
			jmax = j0 > j1 ? j0 : j1;
			jmax = j2 > jmax ? j2 : jmax;

			kmin = k0 < k1 ? k0 : k1;
			kmin = k2 < kmin ? k2 : kmin;
			kmax = k0 > k1 ? k0 : k1;
			kmax = k2 > kmax ? k2 : kmax;

			imin--; jmin--; kmin--;
			imax++; jmax++; kmax++;

		} while ((imax - imin + 1) * (jmax - jmin + 1) * (kmax - kmin + 1) > 150 && level < hash.maxlevel);

		//	ds = hash.ds;

		Triangle3D t3d = Triangle3D(pos[tri[pId][0]], pos[tri[pId][1]], pos[tri[pId][2]]);
		for (int li = imin; li <= imax; li += 1)
			for (int lj = jmin; lj <= jmax; lj += 1)
				for (int lk = kmin; lk <= kmax; lk += 1)
				{
					int i = li, j = lj, k = lk;
					Coord3D ABP11 = Coord3D(i * ds + hash.lo[0] - 0.1 * hash.ds * 10.0,
						j * ds + hash.lo[1] - 0.1 * hash.ds * 10.0,
						k * ds + hash.lo[2] - 0.1 * hash.ds * 10.0);
					Coord3D ABP22 = Coord3D(i * ds + ds + hash.lo[0] + 0.1 * hash.ds * 10.0,
						j * ds + ds + hash.lo[1] + 0.1 * hash.ds * 10.0,
						k * ds + ds + hash.lo[2] + 0.1 * hash.ds * 10.0);
					AlignedBox3D AABB2 = AlignedBox3D(ABP11, ABP22);

					if (AABB2.meshInsert(t3d))
					{
						int gId = hash.getIndex(i, j, k, level);
						if (gId != INVALID)
						{
							int index = atomicAdd(&(hash.counter[gId]), 1);
							hash.ids[hash.index[gId] + index] = -pId - 1;
						}

					}


				}

	}
	template<typename TDataType>
	void GridHash<TDataType>::construct(DeviceArray<Coord>& pos)
	{
		clear();

		dim3 pDims = int(ceil(pos.size() / BLOCK_SIZE + 0.5f));

		K_CalculateParticleNumber << <pDims, BLOCK_SIZE >> > (*this, pos);
		particle_num = m_reduce->accumulate(index, num);

		if (m_scan == nullptr)
		{
			m_scan = new Scan();
		}
		m_scan->exclusive(index, num);

		if (ids != nullptr)
		{
			cuSafeCall(cudaFree(ids));
		}
		cuSafeCall(cudaMalloc((void**)&ids, particle_num * sizeof(int)));

		//		std::cout << "Particle number: " << particle_num << std::endl;

		K_ConstructHashTable << <pDims, BLOCK_SIZE >> > (*this, pos);
		cuSynchronize();
	}


	template<typename TDataType>
	void GridHash<TDataType>::construct(DeviceArray<Coord>& pos, DeviceArray<Triangle>& tri, DeviceArray<Coord>& Tri_pos)
	{
		clear();

		dim3 pDims = int(ceil(pos.size() / BLOCK_SIZE + 0.5f));
		dim3 pDimsTri = int(ceil(tri.size() / BLOCK_SIZE + 0.5f));

		//	K_CalculateParticleNumber << <pDims, BLOCK_SIZE >> > (*this, pos);
		//	cuSynchronize();
		if (!multi_grid)
			K_AddTriNumber << <pDimsTri, BLOCK_SIZE >> > (*this, tri, Tri_pos);
		else
			K_Multi_AddTriNumber << <pDimsTri, BLOCK_SIZE >> > (*this, tri, Tri_pos);

		cuSynchronize();

		particle_num = m_reduce->accumulate(index, num);

		if (m_scan == nullptr)
		{
			m_scan = new Scan();
		}
		m_scan->exclusive(index, num);

		if (ids != nullptr)
		{
			cuSafeCall(cudaFree(ids));
		}
		cuSafeCall(cudaMalloc((void**)&ids, particle_num * sizeof(int)));


		//	K_ConstructHashTable << <pDims, BLOCK_SIZE >> > (*this, pos);
		//	cuSynchronize();
		if (!multi_grid)
			K_AddTriElement << <pDimsTri, BLOCK_SIZE >> > (*this, tri, Tri_pos);
		else
			K_Multi_AddTriElement << <pDimsTri, BLOCK_SIZE >> > (*this, tri, Tri_pos);
		cuSynchronize();
	}

	template<typename TDataType>
	void GridHash<TDataType>::clear()
	{
		cuSafeCall(cudaMemset(counter, 0, num * sizeof(int)));
		cuSafeCall(cudaMemset(index, 0, num * sizeof(int)));
	}

	template<typename TDataType>
	void GridHash<TDataType>::release()
	{
		if (counter != nullptr)
			cuSafeCall(cudaFree(counter));

		if (ids != nullptr)
			cuSafeCall(cudaFree(ids));

		if (index != nullptr)
			cuSafeCall(cudaFree(index));

		// 		if (m_scan != nullptr)
		// 		{
		// 			delete m_scan;
		// 		}
	}
}