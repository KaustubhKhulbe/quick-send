module core
import types::*;
(
    input   logic           clk,
    input   logic           rst

    
);

    cpu_state_t cpu_state, cpu_state_next;


    always_comb begin
        unique case (cpu_state)
            IDLE: begin
                
            end
            READ: begin
                
            end
            WRITE: begin
                
            end
        endcase
    end



    always_ff @(posedge clk) begin
        if (rst) begin
            cpu_state <= IDLE;
        end else begin
            cpu_state <= cpu_state_next;
        end
    end


endmodule


