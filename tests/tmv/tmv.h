

#ifndef _TMV_H_
#define _TMV_H_

// Thread block size
#define BLOCK_SIZE 16

#define MW 2048
// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA MW // Matrix A width
#define HA MW // Matrix A height
#define WB HA // Matrix B width
#define HB 1  // Matrix B H
#define WC WA // Matrix C width
#define HC 1  // Matrix C H

#endif // _TMV_H_

