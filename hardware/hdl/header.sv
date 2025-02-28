module header
import types::*;
(
  input logic clk,
  input logic [3:0][7:0] pixels [31:0],
  output logic compressable,
  output logic [47:0] h
 );

  // header[47:44] are skip bits
  logic [7:0] min_r, min_g, min_b, min_a;

  always_comb begin
    min_r = 8'hFF;
    min_g = 8'hFF;
    min_b = 8'hFF;
    min_a = 8'hFF;
    h = '0;
    compressable = 1'b0;

    for (int i = 0; i < 32; i++) begin
      if (min_r > pixels[i][0]) begin
        min_r = pixels[i][0];
        end

      if (min_g > pixels[i][1]) begin
        min_g = pixels[i][1];
      end

      if (min_b > pixels[i][2]) begin
        min_b = pixels[i][2];
      end

      if (min_a > pixels[i][3]) begin
        min_a = pixels[i][3];
      end
    end

    h[43:36] = min_r;
    h[35:28] = min_g;
    h[27:20] = min_b;
    h[19:12] = min_a;
  end

  endmodule : header
