package types;

    typedef enum logic [3:0] {
        READ,
        WRITE,
        IDLE
    } cpu_state_t;

    typedef struct {
        logic [7:0][3:0] residuals [31:0];
        logic [47:0] header;
        logic        compressable;
    } residual_reg;


endpackage




