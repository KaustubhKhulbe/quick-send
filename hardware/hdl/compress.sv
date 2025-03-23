module compress
(
 input logic clk,
 input logic rst,
 input types::residual_compress_reg cr_reg,
 output types::compress_commit_reg cc_reg
 );

  types::line_t l1;
  logic [2:0] r_req, g_req, b_req, a_req;
  logic [3:0] bit_pos;
  integer index, j;

  always_comb begin
    l1.raw = '0;
    r_req = '0;
    g_req = '0;
    b_req = '0;
    a_req = '0;

    if (cr_reg.compressable) begin
      l1.l.h = cr_reg.header.raw;
      r_req = cr_reg.header.min_values.bits_required[11:9];
      g_req = cr_reg.header.min_values.bits_required[8:6];
      b_req = cr_reg.header.min_values.bits_required[5:3];
      a_req = cr_reg.header.min_values.bits_required[2:0];

      for (int i = 0; i < 32; i++) begin
        bit_pos = 4'b0;
            if (!cr_reg.header.min_values.skip_a) begin
                case (a_req)
                    3'b001: l1.l.pix[i][bit_pos+:2] = cr_reg.residuals.channels.a_channel[i][1:0];
                    3'b000: l1.l.pix[i][bit_pos+:1] = cr_reg.residuals.channels.a_channel[i][0:0];
                    3'b010: l1.l.pix[i][bit_pos+:3] = cr_reg.residuals.channels.a_channel[i][2:0];
                    3'b011: l1.l.pix[i][bit_pos+:4] = cr_reg.residuals.channels.a_channel[i][3:0];
                    3'b100: l1.l.pix[i][bit_pos+:5] = cr_reg.residuals.channels.a_channel[i][4:0];
                    3'b101: l1.l.pix[i][bit_pos+:6] = cr_reg.residuals.channels.a_channel[i][5:0];
                    3'b110: l1.l.pix[i][bit_pos+:7] = cr_reg.residuals.channels.a_channel[i][6:0];
                    3'b111: l1.l.pix[i][bit_pos+:8] = cr_reg.residuals.channels.a_channel[i][7:0];
                endcase

                bit_pos += 4'(a_req + 1);
            end

            if (!cr_reg.header.min_values.skip_b) begin
                case (b_req)
                    3'b000: l1.l.pix[i][bit_pos+:1] = cr_reg.residuals.channels.b_channel[i][0:0];
                    3'b001: l1.l.pix[i][bit_pos+:2] = cr_reg.residuals.channels.b_channel[i][1:0];
                    3'b010: l1.l.pix[i][bit_pos+:3] = cr_reg.residuals.channels.b_channel[i][2:0];
                    3'b011: l1.l.pix[i][bit_pos+:4] = cr_reg.residuals.channels.b_channel[i][3:0];
                    3'b100: l1.l.pix[i][bit_pos+:5] = cr_reg.residuals.channels.b_channel[i][4:0];
                    3'b101: l1.l.pix[i][bit_pos+:6] = cr_reg.residuals.channels.b_channel[i][5:0];
                    3'b110: l1.l.pix[i][bit_pos+:7] = cr_reg.residuals.channels.b_channel[i][6:0];
                    3'b111: l1.l.pix[i][bit_pos+:8] = cr_reg.residuals.channels.b_channel[i][7:0];
                endcase

                bit_pos += 4'(b_req + 1);
            end

            if (!cr_reg.header.min_values.skip_g) begin
                case (g_req)
                    3'b000: l1.l.pix[i][bit_pos+:1] = cr_reg.residuals.channels.g_channel[i][0:0];
                    3'b001: l1.l.pix[i][bit_pos+:2] = cr_reg.residuals.channels.g_channel[i][1:0];
                    3'b010: l1.l.pix[i][bit_pos+:3] = cr_reg.residuals.channels.g_channel[i][2:0];
                    3'b011: l1.l.pix[i][bit_pos+:4] = cr_reg.residuals.channels.g_channel[i][3:0];
                    3'b100: l1.l.pix[i][bit_pos+:5] = cr_reg.residuals.channels.g_channel[i][4:0];
                    3'b101: l1.l.pix[i][bit_pos+:6] = cr_reg.residuals.channels.g_channel[i][5:0];
                    3'b110: l1.l.pix[i][bit_pos+:7] = cr_reg.residuals.channels.g_channel[i][6:0];
                    3'b111: l1.l.pix[i][bit_pos+:8] = cr_reg.residuals.channels.g_channel[i][7:0];
                endcase

                bit_pos += 4'(g_req + 1);
            end

            if (!cr_reg.header.min_values.skip_r) begin
                case (r_req)
                    3'b000: l1.l.pix[i][bit_pos+:1] = cr_reg.residuals.channels.r_channel[i][0:0];
                    3'b001: l1.l.pix[i][bit_pos+:2] = cr_reg.residuals.channels.r_channel[i][1:0];
                    3'b010: l1.l.pix[i][bit_pos+:3] = cr_reg.residuals.channels.r_channel[i][2:0];
                    3'b011: l1.l.pix[i][bit_pos+:4] = cr_reg.residuals.channels.r_channel[i][3:0];
                    3'b100: l1.l.pix[i][bit_pos+:5] = cr_reg.residuals.channels.r_channel[i][4:0];
                    3'b101: l1.l.pix[i][bit_pos+:6] = cr_reg.residuals.channels.r_channel[i][5:0];
                    3'b110: l1.l.pix[i][bit_pos+:7] = cr_reg.residuals.channels.r_channel[i][6:0];
                    3'b111: l1.l.pix[i][bit_pos+:8] = cr_reg.residuals.channels.r_channel[i][7:0];
                endcase

                bit_pos += 4'(r_req + 1);
            end

        end

    end
  end

  always_comb begin
    cc_reg.compressable = cr_reg.compressable;
    cc_reg.flag = '0;
    cc_reg.lines.l1 = '0;
    cc_reg.lines.l2 = '0;
    if (cr_reg.compressable) begin
        cc_reg.lines.l1 = l1;
        cc_reg.lines.l2.raw = '0;
        cc_reg.flag = 2'b01;
    end
    end

endmodule : compress
