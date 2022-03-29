
#ifndef _VV_H_
#define _VV_H_

// Thread block size
#define BLOCK_SIZE 16

#define MW 2048
// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA MW // Matrix A width
#define HA 1 // Matrix A height
#define WB MW // Matrix B width
#define HB 1  // Matrix B H
#define WC MW // Matrix C width
#define HC MW  // Matrix C H

#endif // _VV_H_

