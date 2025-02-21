module cpu
(
    input   logic           clk,
    input   logic           rst,

    output  logic           [15:0] tmp
);
    always_ff @(posedge clk) begin
      if (rst) begin
        tmp <= '0;
      end
      else begin
        tmp <= '1;
        end
    end


endmodule


