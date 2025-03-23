package types;

    typedef enum logic [3:0] {
        READ,
        WRITE,
        IDLE
    } cpu_state_t;

  typedef union packed {
    logic [47:0] raw;  // Raw 48-bit header representation
    struct packed {
      logic skip_r;
      logic skip_g;
      logic skip_b;
      logic skip_a;
      logic [7:0] r_min;
      logic [7:0] g_min;
      logic [7:0] b_min;
      logic [7:0] a_min;
      logic [11:0] bits_required; // todo: change to real name
      } min_values;
    } header_t;

  typedef union packed {
    struct packed {
      logic [31:0][7:0] r_channel;
      logic [31:0][7:0] g_channel;
      logic [31:0][7:0] b_channel;
      logic [31:0][7:0] a_channel;
    } channels;

    logic [31:0] [3:0] [7:0] pixels;
  } pixels_t;

  typedef union packed {
    logic [64 * 8 - 1 : 0] raw;
    struct packed {
      header_t h;
      logic [0:31][13:0] pix;
      logic [15:0]       pad;
    } l;
    } line_t;

  typedef struct {
    pixels_t pixels;
    header_t header;
    logic compressable;
    struct packed {
      logic [7:0] r_max;
      logic [7:0] g_max;
      logic [7:0] b_max;
      logic [7:0] a_max;
      } max_pixels;
    } header_residual_reg;

  typedef struct {
    pixels_t residuals;
    header_t header;
    logic compressable;
    } residual_compress_reg;

  typedef struct {
    logic compressable;
    logic [1:0] flag;
    struct packed {
      line_t l1;
      line_t l2;
    } lines;
  } compress_commit_reg;

endpackage
