module residual
import types::*;
(
  input types::header_residual_reg hr_reg,
  output types::residual_compress_reg cr_reg
 );

  logic [3:0] [7:0] rgba_logs;
  types::pixels_t sub_pixels;

  logic [7:0] r_diff, g_diff, b_diff, a_diff;

  types::header_t header;

  always_comb begin
    r_diff = hr_reg.max_pixels.r_max - hr_reg.header.min_values.r_min;
    g_diff = hr_reg.max_pixels.g_max - hr_reg.header.min_values.g_min;
    b_diff = hr_reg.max_pixels.b_max - hr_reg.header.min_values.b_min;
    a_diff = hr_reg.max_pixels.a_max - hr_reg.header.min_values.a_min;

    rgba_logs[0] = (r_diff == 8'd0) ? 8'd0 :
                   (r_diff <= 8'd1) ? 8'd1 :
                   (r_diff <= 8'd3) ? 8'd2 :
                   (r_diff <= 8'd7) ? 8'd3 :
                   (r_diff <= 8'd15) ? 8'd4 :
                   (r_diff <= 8'd31) ? 8'd5 :
                   (r_diff <= 8'd63) ? 8'd6 :
                   (r_diff <= 8'd127) ? 8'd7 :
                   8'd8;

    rgba_logs[1] = (g_diff == 8'd0) ? 8'd0 :
                   (g_diff <= 8'd1) ? 8'd1 :
                   (g_diff <= 8'd3) ? 8'd2 :
                   (g_diff <= 8'd7) ? 8'd3 :
                   (g_diff <= 8'd15) ? 8'd4 :
                   (g_diff <= 8'd31) ? 8'd5 :
                   (g_diff <= 8'd63) ? 8'd6 :
                   (g_diff <= 8'd127) ? 8'd7 :
                   8'd8;

    rgba_logs[2] = (b_diff == 8'd0) ? 8'd0 :
                   (b_diff <= 8'd1) ? 8'd1 :
                   (b_diff <= 8'd3) ? 8'd2 :
                   (b_diff <= 8'd7) ? 8'd3 :
                   (b_diff <= 8'd15) ? 8'd4 :
                   (b_diff <= 8'd31) ? 8'd5 :
                   (b_diff <= 8'd63) ? 8'd6 :
                   (b_diff <= 8'd127) ? 8'd7 :
                   8'd8;

    rgba_logs[3] = (a_diff == 8'd0) ? 8'd0 :
                   (a_diff <= 8'd1) ? 8'd1 :
                   (a_diff <= 8'd3) ? 8'd2 :
                   (a_diff <= 8'd7) ? 8'd3 :
                   (a_diff <= 8'd15) ? 8'd4 :
                   (a_diff <= 8'd31) ? 8'd5 :
                   (a_diff <= 8'd63) ? 8'd6 :
                   (a_diff <= 8'd127) ? 8'd7 :
                   8'd8;

    header.min_values.skip_r = (r_diff == '0) ? 1'b1 : 1'b0;
    header.min_values.skip_g = (g_diff == '0) ? 1'b1 : 1'b0;
    header.min_values.skip_b = (b_diff == '0) ? 1'b1 : 1'b0;
    header.min_values.skip_a = (a_diff == '0) ? 1'b1 : 1'b0;
    header.min_values.r_min = hr_reg.header.min_values.r_min;
    header.min_values.g_min = hr_reg.header.min_values.g_min;
    header.min_values.b_min = hr_reg.header.min_values.b_min;
    header.min_values.a_min = hr_reg.header.min_values.a_min;
    header.min_values.bits_required[11:9] = (r_diff != '0) ? 3'(rgba_logs[0] - 1) : 3'b0;
    header.min_values.bits_required[8:6]  = (g_diff != '0) ? 3'(rgba_logs[1] - 1) : 3'b0;
    header.min_values.bits_required[5:3]  = (b_diff != '0) ? 3'(rgba_logs[2] - 1) : 3'b0;
    header.min_values.bits_required[2:0]  = (a_diff != '0) ? 3'(rgba_logs[3] - 1) : 3'b0;

    for (int i = 0; i < 32; i++) begin
      sub_pixels.pixels[i][0] = hr_reg.pixels.pixels[i][0] - hr_reg.header.min_values.r_min;
      sub_pixels.pixels[i][1] = hr_reg.pixels.pixels[i][1] - hr_reg.header.min_values.g_min;
      sub_pixels.pixels[i][2] = hr_reg.pixels.pixels[i][2] - hr_reg.header.min_values.b_min;
      sub_pixels.pixels[i][3] = hr_reg.pixels.pixels[i][3] - hr_reg.header.min_values.a_min;
    end
    cr_reg.compressable = ((rgba_logs[0] + rgba_logs[1] + rgba_logs[2] + rgba_logs[3]) <= 14) ? 1'b1 : 1'b0;
    cr_reg.residuals = sub_pixels;
    cr_reg.header = header;
  end

  endmodule : residual
