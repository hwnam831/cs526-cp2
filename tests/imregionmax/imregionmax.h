#ifndef _IMREGIONMAX_H_
#define _IMREGIONMAX_H_

// Thread block size
#define BLOCK_SIZEX 1
#define BLOCK_SIZEY 1

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA 2048 // Matrix A width
#define HA 2048 // Matrix A height

#define WIDTH_A (15+WA+1)  // Matrix A height after padding
#define HEIGHT_A (15+HA+1) // Matrix A height after padding

#define WIDTH_C WA // Matrix C width
#define HEIGHT_C HA // Matrix C height

#endif // _IMREGIONMAX_H_
