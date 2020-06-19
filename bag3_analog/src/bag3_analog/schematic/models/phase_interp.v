# SPDX-License-Identifier: Apache-2.0
# Copyright 2020 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
{{ _header }}
    // The phase interpolator behavioral model behaves as follows:
    //
    // Normal (expected) operation:
    //     1. Wait for a_in or b_in to toggle. Register the one that toggled as early
    //     2. Wait for the other signal to toggle. Register this one as late.
    //        Under normal operations, it'll be the same direction as the early (e.g., a rise b rise).
    //     3. Measure the delay between the two inputs.
    //     4. Based on the en/enb settings for the two signals and the relative delay steps,
    //        calculate the relative delay (i.e., as a proportion to the delay between the two inputs).
    //     5. Calculate the absolute delay, accounting for the intrinsic delay as well.
    //     6. Propagate the inputs to the output with the absolute delay.
    //
    // This step-by-step operation is enabled by the FSM, which has the following states:
    //     IDLE        : Waiting for an early edge.
    //     A_EARLY_RISE: Waiting for late edge given that early edge was rising a_in.
    //                   Under normal operation, late edge then should be rising b_in.
    //     A_EARLY_FALL: Waiting for late edge given that early edge was falling a_in.
    //                   Under normal operation, late edge then should be falling b_in.
    //     B_EARLY_RISE: Waiting for late edge given that early edge was rising b_in.
    //                   Under normal operation, late edge then should be rising a_in.
    //     B_EARLY_FALL: Waiting for late edge given that early edge was falling b_in.
    //                   Under normal operation, late edge then should be falling a_in.
    //
    // The delay calculation and delayed assignment to output happens when the late signal is registered,
    // which happens at the state transition from A/B_EARLY_RISE/FALL to IDLE.
    //
    // Edge cases:
    //     If any of the inputs and enable signals is x, then output is x immediately.
    //
    //     If a_en and a_enb are not complementary, then error. Same for b_en and b_enb. Same for a_en and b_en.
    //
    //     If the intrinsic delay less than the calculated delay (which is a linear function of the
    //      delay between the 2 inputs) such that the delay from the late input to the output is negative,
    //      then error and don't update the output.
    //      Note: this isn't an actual "problem" in the actual circuit, but the model works by
    //      applying a delay relative to the late edge. Hence it won't work in this case.
    //
    //     If the early edge is rising and the late edge is falling (or vice versa), then error.
    //      However, treat the late falling edge as the early falling edge and update the state correspondingly.
    //      In other words, assume the early edge was invalid (or that it was actually a late edge and the
    //      previous rising edge was not captured).
    //
    //     If one input toggles (and is detected as the early edge) and the same input toggles again
    //      without the other input toggling, then error. However, treat the latter toggle as the early
    //      edge, update the state correspondingly, and proceed. In other words, assume the former edge was invalid.
    //
    //     When an input is x, output x immediately as described above. In addition, do not update the state.
    //      Note that this means that the FSM can see two consecutive (defined) toggles from the same input
    //      in the same direction. For example, a_in can be 0 -> 1 -> x -> 1. In this case, as was with
    //      the above cases, error but treat the latter rising edge (x -> 1) as the early rising edge.
    //      In other words, assume that the former rising edge (0 -> 1) was invalid.
    //
    //     If both inputs change simultaneously, the model interprets them given the current state:
    //      IDLE        : If both signals rise or fall, then arbitrarily assign one as early and the other as late.
    //                      Proceed to delay calculation. Keep state at IDLE.
    //                    If one rises and the other falls, error and keep state at IDLE.
    //                    If one rises or falls and the other becomes x, ignore the latter and register the
    //                      former as the early edge.
    //                    If both become x, error, ignore both, and keep the state at IDLE.
    //      A_EARLY_RISE: If a_in falls and b_in rises, associate rising b_in with the previously detected
    //                      a_in and register it as the late edge. Proceed to delay calculation accordingly.
    //                      Register the falling a_in as the early edge for the next pair of edges and thus
    //                      advance to A_EARLY_FALL.
    //                    If both rise, ignore the previously detected early edge and use the recent rising edges.
    //                      Arbitrarily assign one as early and the other as late. Proceed to delay calculation.
    //                      Advance to IDLE. Note that this only happens when a_in was 1 -> x before and
    //                      both a_in and b_in transition to 1.
    //                    If both fall, error because rising b_in was expected. Ignore the previously detected
    //                      early edge and use the recent falling edges. Arbitrarily assign one as early
    //                      and the other as late. Proceed to delay calculation. Advance to IDLE.
    //                    If a_in rises and b_in falls, error and advance to IDLE. Note that this only
    //                      happens when a_in was 1 -> x before. and a_in transitions to 1, while b_in falls.
    //                    If either a_in or b_in becomes x, ignore the x and treat it as a single edge toggling.
    //                    If both become x, error and keep state at A_EARLY_RISE.
    //     All the other states are mirrored cases of A_EARLY_RISE.

    time intrinsic = {{ intrinsic }};  // intrinsic delay of circuit
    parameter ASSERTION_DELAY = {{ assertion_delay | default(0, true) }}; // Delay for assertion checkers

    logic late_in;
    logic [{{ _sch_params['nbits'] - 1 }}:0] late_en;
    time early_time, late_time, late_early_delta;

    real delay_scalar;                // relative delay from late edge to output in
                                      // proportion to delay between early and late edge
    time calc_delay, neg_calc_delay;  // neg_calc_delay is used as a sanity check to
                                      // ensure that the intrinsic delay is large enough
                                      // for the model to work properly

    logic temp;

    // FSM states
    typedef enum {IDLE, A_EARLY_RISE, A_EARLY_FALL, B_EARLY_RISE, B_EARLY_FALL} state_t;
    state_t curr_state;

    logic prev_a_in, prev_b_in;
    logic a_stable, a_new_x, a_rise, a_fall;
    logic b_stable, b_new_x, b_rise, b_fall;
    time next_early_time;  // only used when early_time must be used to calculate
                           // the delay but the next early edge is already detected

    wire has_inputs_x;
    assign has_inputs_x = $isunknown({a_en, a_enb, b_en, b_enb, a_in, b_in});

    always @(posedge a_in or posedge b_in or negedge a_in or negedge b_in) begin
        a_stable = a_in === prev_a_in;
        b_stable = b_in === prev_b_in;
        a_new_x = $isunknown(a_in) && !$isunknown(prev_a_in);
        b_new_x = $isunknown(b_in) && !$isunknown(prev_b_in);
        a_rise = (!$isunknown(a_in) && a_in)  && ($isunknown(prev_a_in) || !prev_a_in) && !a_stable;
        a_fall = (!$isunknown(a_in) && !a_in) && ($isunknown(prev_a_in) || prev_a_in)  && !a_stable;
        b_rise = (!$isunknown(b_in) && b_in)  && ($isunknown(prev_b_in) || !prev_b_in) && !b_stable;
        b_fall = (!$isunknown(b_in) && !b_in) && ($isunknown(prev_b_in) || prev_b_in)  && !b_stable;

        if ((a_new_x && $isunknown(b_in)) || ($isunknown(a_in) && b_new_x)) begin
            `ifdef PI_ERROR
                $error("Invalid transition because a_in = b_in = x. Not updating curr_state");
            `else
                $info("Invalid transition because a_in = b_in = x. Not updating curr_state");
            `endif
        end
        else begin
            case (curr_state)
                IDLE: begin
                    early_time = $time;
                    casez({a_rise, a_fall, b_rise, b_fall})
                        4'b10_10, 4'b01_01: begin
                            curr_state <= IDLE;
                            late_time = $time;
                            // Doesn't matter which is early and late if both arrive simultaneously
                            late_in <= b_in;
                            late_en <= b_en;
                        end
                        4'b10_01, 4'b01_10: begin
                            `ifdef PI_ERROR
                                $error("a_in and b_in changed simultaneously but in reverse polarity");
                            `else
                                $info("a_in and b_in changed simultaneously but in reverse polarity");
                            `endif
                            curr_state <= IDLE;
                        end
                        4'b10_00, 4'b01_00: begin
                            curr_state <= a_rise ? A_EARLY_RISE : A_EARLY_FALL;
                        end
                        4'b00_10, 4'b00_01: begin
                            curr_state <= b_rise ? B_EARLY_RISE : B_EARLY_FALL;
                        end
                        4'b00_00: begin
                            if (a_new_x)
                                `ifdef PI_ERROR
                                    $error("Invalid transition because a_in = x. Not updating curr_state");
                                `else
                                    $info("Invalid transition because a_in = x. Not updating curr_state");
                                `endif
                            else if (b_new_x)
                                `ifdef PI_ERROR
                                    $error("Invalid transition because b_in = x. Not updating curr_state");
                                `else
                                    $info("Invalid transition because b_in = x. Not updating curr_state");
                                `endif
                        end
                    endcase
                end
                {#
                Note: the FSM behavior for A_EARLY_RISE, A_EARLY_FALL, B_EARLY_RISE, and B_EARLY_FALL
                all mirror each other. Instead of writing out each part individually (which would make
                the code long and harder to maintain if a small part needs to be changed for all these
                states), the following Jinja code will write it out for all 4 states.
                However, this makes the following code a bit hard to read, so please look at the
                generated behavioral model to get a better understanding of what this code prints.
                #}
                {% for sig in ['a', 'b'] %}
                {% for edge in ['rise', 'fall'] %}
                {% set state = sig.upper() + '_EARLY_' + edge.upper() %}
                {% set other_sig = 'b' if sig == 'a' else 'a' %}
                {% set other_edge = 'fall' if edge == 'rise' else 'rise' %}
                {% set state_other_edge = sig.upper() + '_EARLY_' + other_edge.upper() %}
                {% set state_other_sig_edge = other_sig.upper() + '_EARLY_' + other_edge.upper() %}
                {% set case_inputs = [sig       + '_' + edge,
                                      sig       + '_' + other_edge,
                                      other_sig + '_' + edge,
                                      other_sig + '_' + other_edge] %}
                {% set case_inputs = '{' + ', '.join(case_inputs) + '}' %}
                {{ state }}: begin
                    casez({{ case_inputs }})
                        4'b01_10: begin
                            curr_state <= {{ state_other_edge }};
                            late_time = $time;
                            next_early_time = $time;
                            late_in <= {{ other_sig }}_in;
                            late_en <= {{ other_sig }}_en;
                        end
                        4'b10_10, 4'b01_01: begin
                            if ({{ sig }}_{{ edge }})
                                `ifdef PI_ERROR
                                    $error("Expected negedge, received posedge instead");
                                `else
                                    $info("Expected negedge, received posedge instead");
                                `endif
                            else
                                `ifdef PI_ERROR
                                    $error("Expected posedge, received negedge instead");
                                `else
                                    $info("Expected posedge, received negedge instead");
                                `endif
                            curr_state <= IDLE;
                            early_time = $time;
                            late_time = $time;
                            // Doesn't matter which is early and late if both arrive simultaneously
                            late_in <= b_in;
                            late_en <= b_en;
                        end
                        4'b10_01: begin
                            `ifdef PI_ERROR
                                $error("a_in and b_in changed simultaneously but in reverse polarity");
                            `else
                                $info("a_in and b_in changed simultaneously but in reverse polarity");
                            `endif
                            curr_state <= IDLE;
                        end
                        4'b10_00, 4'b01_00: begin
                            `ifdef PI_ERROR
                                $error("Expected {{ other_sig }}_in to change, not {{ sig }}_in");
                            `else
                                $info("Expected {{ other_sig }}_in to change, not {{ sig }}_in");
                            `endif
                            curr_state <= {{sig}}_{{edge}} ? {{ state }} : {{ state_other_edge }};
                            early_time = $time;
                        end
                        4'b00_01: begin
                            `ifdef PI_ERROR
                                $error("Expected posedge, received negedge instead");
                            `else
                                $info("Expected posedge, received negedge instead");
                            `endif
                            curr_state <= {{ state_other_sig_edge }};
                            early_time = $time;
                        end
                        4'b00_10: begin
                            curr_state <= IDLE;
                            late_time = $time;
                            late_in <= {{ other_sig }}_in;
                            late_en <= {{ other_sig }}_en;
                        end
                        default: begin
                            if (a_new_x)
                                `ifdef PI_ERROR
                                    $error("Invalid transition because a_in = x. Not updating curr_state");
                                `else
                                    $info("Invalid transition because a_in = x. Not updating curr_state");
                                `endif
                            else if (b_new_x)
                                `ifdef PI_ERROR
                                    $error("Invalid transition because b_in = x. Not updating curr_state");
                                `else
                                    $info("Invalid transition because b_in = x. Not updating curr_state");
                                `endif
                        end
                    endcase
                end
                {% endfor %}
                {% endfor %}
            endcase
        end

        prev_a_in <= a_in;
        prev_b_in <= b_in;
    end

    always @(late_in) begin
        // Calculate delay_scalar for each thermometer code
        case ($countones(late_en))
            {% for step_val in steps %}
            {{ loop.index - 1 }}: delay_scalar = {{ step_val }};
            {% endfor %}
        endcase

        late_early_delta = late_time - early_time;
        calc_delay = intrinsic - (1 - delay_scalar) * late_early_delta;
        neg_calc_delay = (1 - delay_scalar) * late_early_delta - intrinsic;

        if (intrinsic > (1 - delay_scalar) * late_early_delta)
            temp = #(calc_delay) late_in;
        else
            `ifdef PI_ERROR
                $error("intrinsic delay (%0t) is too small, resulting in negative calc_delay (-%0t)", intrinsic, neg_calc_delay);
            `else
                $info("intrinsic delay (%0t) is too small, resulting in negative calc_delay (-%0t)", intrinsic, neg_calc_delay);
            `endif

        early_time <= next_early_time;
    end

    // enable complement assertion checker
    // With real delays, no two signals will change at the same time. Thus, only check for complementary signals
    // after ASSERTION_DELAY, which is expected to be enough time for all relevant signals to change
    {% set comp_sigs = [('a_en', 'a_enb'), ('b_en', 'b_enb'), ('a_en', 'b_en')] %}
    {% for sig1, sig2 in comp_sigs %}
    always @(*) begin
        if (!$isunknown({ {{ sig1 }}, {{ sig2 }} })) begin
            #(ASSERTION_DELAY);
            assert ({{ sig1 }} == ~{{ sig2 }}) else
                `ifdef PI_ERROR
                    $error("{{ sig1 }} (%b) and {{ sig2 }} (%b) must be complementary", {{ sig1 }}, {{ sig2 }});
                `else
                    $info("{{ sig1 }} (%b) and {{ sig2 }} (%b) must be complementary", {{ sig1 }}, {{ sig2 }});
                `endif
        end
    end
    {% endfor %}

    // enable x assertion checker
    {% for sig in ['a_en', 'a_enb', 'b_en', 'b_enb'] %}
    always_comb begin
        assert (!$isunknown({{ sig }})) else
            `ifdef PI_ERROR
                $error("{{ sig }} (%b) contains x's", {{ sig }});
            `else
                $info("{{ sig }} (%b) contains x's", {{ sig }});
            `endif
    end
    {% endfor %}

    assign out = has_inputs_x ? 1'bx : temp;

endmodule
