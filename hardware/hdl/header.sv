module header
import types::*;
(
 input logic clk,
 input logic rst,
 input types::pixels_t pixels,
 output types::header_residual_reg hr_reg
 );

  logic [3:0] [7:0] mins;
  logic [5:0] [3:0] [7:0] imms;

  always_comb begin
    imms[0] = '1;
    for (int i = 0; i < 8; i++) begin
      for (int j = 0; j < 4; j++) begin
        if (pixels.pixels[i][j] < imms[0][j]) imms[0][j] = pixels.pixels[i][j];
        end
      end
    end
  always_comb begin
    imms[1] = '1;
    for (int i = 8; i < 16; i++) begin
      for (int j = 0; j < 4; j++) begin
        if (pixels.pixels[i][j] < imms[1][j]) imms[1][j] = pixels.pixels[i][j];
        end
      end
    end

  always_comb begin
    imms[2] = '1;
    for (int i = 16; i < 24; i++) begin
      for (int j = 0; j < 4; j++) begin
        if (pixels.pixels[i][j] < imms[2][j]) imms[2][j] = pixels.pixels[i][j];
        end
      end
    end

  always_comb begin
    imms[3] = '1;
    for (int i = 24; i < 32; i++) begin
      for (int j = 0; j < 4; j++) begin
        if (pixels.pixels[i][j] < imms[3][j]) imms[3][j] = pixels.pixels[i][j];
        end
      end
    end

  always_comb begin
    for (int j = 0; j < 4; j++) begin
      imms[4][j] = (imms[0][j] < imms[1][j]) ? imms[0][j] : imms[1][j];
      end
    end

  always_comb begin
    for (int j = 0; j < 4; j++) begin
      imms[5][j] = (imms[2][j] < imms[3][j]) ? imms[2][j] : imms[3][j];
      end
    end

  always_ff @(posedge clk) begin
    hr_reg.pixels = pixels;
    if (rst) begin
      hr_reg.compressable = 1'b0;
      hr_reg.header.raw = '0;
      end else begin

      hr_reg.compressable = 1'b1;

      if (imms[4][0] < imms[5][0]) hr_reg.header.min_values.r_min = imms[4][0];
      else hr_reg.header.min_values.r_min = imms[5][0];

      if (imms[4][1] < imms[5][1]) hr_reg.header.min_values.g_min = imms[4][1];
      else hr_reg.header.min_values.g_min = imms[5][1];

      if (imms[4][2] < imms[5][2]) hr_reg.header.min_values.b_min = imms[4][2];
      else hr_reg.header.min_values.b_min = imms[5][2];

      if (imms[4][3] < imms[5][3]) hr_reg.header.min_values.a_min = imms[4][3];
      else hr_reg.header.min_values.a_min = imms[5][3];
      end
    end

  endmodule : header
