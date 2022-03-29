/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include <stdio.h>
#include <math.h>
#include <float.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" void computeGold(float* a, float* b, const unsigned int len,
		float* result);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void computeGold(float* a, float* b, const unsigned int len, float* result) {
	for (unsigned int i = 0; i < len; i++) {
		for (unsigned int j = 0; j < len; j++) {
			result[j * len + i] += a[j] * b[i];
		}
	}

}

void swaprows(float** arr, long row0, long row1) {
	float* temp;
	temp = arr[row0];
	arr[row0] = arr[row1];
	arr[row1] = temp;
}
//        gjelim
void gjelim(float** lhs, float** rhs, long nrows, long ncolsrhs) {
	//        augment lhs array with rhs array and store in arr2
	float** arr2 = new float*[nrows];
	for (long row = 0; row < nrows; ++row)
		arr2[row] = new float[nrows + ncolsrhs];
	for (long row = 0; row < nrows; ++row) {
		for (long col = 0; col < nrows; ++col) {
			arr2[row][col] = lhs[row][col];
		}
		for (long col = nrows; col < nrows + ncolsrhs; ++col) {
			arr2[row][col] = rhs[row][col - nrows];
		}
	}
	//        perform forward elimination to get arr2 in row-echelon form
	for (long dindex = 0; dindex < nrows; ++dindex) {
		//        run along diagonal, swapping rows to move zeros in working position
		//        (along the diagonal) downwards
		if ((dindex == (nrows - 1)) && (arr2[dindex][dindex] == 0)) {
			return; //  no solution
		} else if (arr2[dindex][dindex] == 0) {
			swaprows(arr2, dindex, dindex + 1);
		}
		//        divide working row by value of working position to get a 1 on the
		//        diagonal
		if (arr2[dindex][dindex] == 0.0) {
			return;
		} else {
			float tempval = arr2[dindex][dindex];
			for (long col = 0; col < nrows + ncolsrhs; ++col) {
				arr2[dindex][col] /= tempval;
			}
		}
		//        eliminate value below working position by subtracting a multiple of
		//        the current row
		for (long row = dindex + 1; row < nrows; ++row) {
			float wval = arr2[row][dindex];
			for (long col = 0; col < nrows + ncolsrhs; ++col) {
				arr2[row][col] -= wval * arr2[dindex][col];
			}
		}
	}
	//        backward substitution steps
	for (long dindex = nrows - 1; dindex >= 0; --dindex) {
		//        eliminate value above working position by subtracting a multiple of
		//        the current row
		for (long row = dindex - 1; row >= 0; --row) {
			float wval = arr2[row][dindex];
			for (long col = 0; col < nrows + ncolsrhs; ++col) {
				arr2[row][col] -= wval * arr2[dindex][col];
			}
		}
	}
	//        assign result to replace rhs
	for (long row = 0; row < nrows; ++row) {
		for (long col = 0; col < ncolsrhs; ++col) {
			rhs[row][col] = arr2[row][col + nrows];
		}
	}
	for (long row = 0; row < nrows; ++row)
		delete[] arr2[row];
	delete[] arr2;
}
