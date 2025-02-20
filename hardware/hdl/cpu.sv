module cpu
(
    input   logic           clk,
    input   logic           rst,

    input   logic   [31:0]  lookup_pc
);

    bit tmp;
    always_ff @(posedge clk) begin
        if (rst) begin
           if (lookup_pc == 32'h00000000) begin
                tmp <= 1'b1;
           end
        end
    end


    
endmodule


