module header
import types::*;
(
 input logic clk,
 input logic rst,
 input types::pixels_t pixels,
 output types::header_residual_reg hr_reg
 );

  logic [3:0] [7:0] mins;
  logic [5:0] [3:0] [7:0] imms_min;

  logic [3:0] [7:0] maxes;
  logic [5:0] [3:0] [7:0] imms_max;

  function automatic logic [3:0] [7:0] find_min(input int start_idx, input int end_idx);
  logic [3:0] [7:0] min_val;
  min_val = '1;
  for (int i = start_idx; i < end_idx; i++) begin
  for (int j = 0; j < 4; j++) begin
      if (pixels.pixels[i][j] < min_val[j])
          min_val[j] = pixels.pixels[i][j];
  end
  end
  return min_val;
  endfunction; // min

  function automatic logic [3:0] [7:0] find_max(input int start_idx, input int end_idx);
  logic [3:0] [7:0] max_val;
  max_val = '0;
  for (int i = start_idx; i < end_idx; i++) begin
  for (int j = 0; j < 4; j++) begin
      if (pixels.pixels[i][j] > max_val[j])
          max_val[j] = pixels.pixels[i][j];
  end
  end
  return max_val;
  endfunction; // max


  always_comb imms_min[0] = find_min(0, 8);
  always_comb imms_min[1] = find_min(8, 16);
  always_comb imms_min[2] = find_min(16, 24);
  always_comb imms_min[3] = find_min(24, 32);

  always_comb imms_max[0] = find_max(0, 8);
  always_comb imms_max[1] = find_max(8, 16);
  always_comb imms_max[2] = find_max(16, 24);
  always_comb imms_max[3] = find_max(24, 32);

  always_comb begin

    end

  always_comb begin
    for (int j = 0; j < 4; j++) begin
      imms_min[4][j] = (imms_min[0][j] < imms_min[1][j]) ? imms_min[0][j] : imms_min[1][j];
      imms_min[5][j] = (imms_min[2][j] < imms_min[3][j]) ? imms_min[2][j] : imms_min[3][j];
    end

    mins[0] = (imms_min[4][0] < imms_min[5][0]) ? imms_min[4][0] : imms_min[5][0];
    mins[1] = (imms_min[4][1] < imms_min[5][1]) ? imms_min[4][1] : imms_min[5][1];
    mins[2] = (imms_min[4][2] < imms_min[5][2]) ? imms_min[4][2] : imms_min[5][2];
    mins[3] = (imms_min[4][3] < imms_min[5][3]) ? imms_min[4][3] : imms_min[5][3];

  end

  always_comb begin
    for (int j = 0; j < 4; j++) begin
      imms_max[4][j] = (imms_max[0][j] > imms_max[1][j]) ? imms_max[0][j] : imms_max[1][j];
      imms_max[5][j] = (imms_max[2][j] > imms_max[3][j]) ? imms_max[2][j] : imms_max[3][j];
    end

    maxes[0] = (imms_max[4][0] > imms_max[5][0]) ? imms_max[4][0] : imms_max[5][0];
    maxes[1] = (imms_max[4][1] > imms_max[5][1]) ? imms_max[4][1] : imms_max[5][1];
    maxes[2] = (imms_max[4][2] > imms_max[5][2]) ? imms_max[4][2] : imms_max[5][2];
    maxes[3] = (imms_max[4][3] > imms_max[5][3]) ? imms_max[4][3] : imms_max[5][3];

  end

  always_ff @(posedge clk) begin
    hr_reg.compressable <= 1'b1;
    hr_reg.pixels <= pixels;
    hr_reg.header.min_values.bits_required <= '0;
    hr_reg.header.min_values.skip_r <= 1'b0;
    hr_reg.header.min_values.skip_g <= 1'b0;
    hr_reg.header.min_values.skip_b <= 1'b0;
    hr_reg.header.min_values.skip_a <= 1'b0;

    if (rst) begin
      hr_reg.compressable <= 1'b0;
      hr_reg.pixels.channels.r_channel <= '0;
      hr_reg.pixels.channels.g_channel <= '0;
      hr_reg.pixels.channels.b_channel <= '0;
      hr_reg.pixels.channels.a_channel <= '0;
    end else begin
      hr_reg.header.min_values.r_min <= mins[0];
      hr_reg.header.min_values.g_min <= mins[1];
      hr_reg.header.min_values.b_min <= mins[2];
      hr_reg.header.min_values.a_min <= mins[3];

      hr_reg.max_pixels.r_max <= maxes[0];
      hr_reg.max_pixels.g_max <= maxes[1];
      hr_reg.max_pixels.b_max <= maxes[2];
      hr_reg.max_pixels.a_max <= maxes[3];
    end
  end

  endmodule : header
