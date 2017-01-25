# tests for strategy functions.

using Base.Test

push!(LOAD_PATH, ".")
include("includes.jl")

function get_test_state()
    uav_state_own = UAVState(100., -10., 0., 0., 1., 0., 0., 0.)
    uav_state_int = UAVState(0., 10., 0., 0., 1., 0., 0., 0.)
    s = State([uav_state_own; uav_state_int])
    s.responding_states = [1,1]
    return s
end

function test_modify_utilities()
    uav = get_real_uav()
    state = get_test_state()
    # second signal doesn't matter here
    state.signals = [:dont_turn_left, :dont_turn_right]
    if uav.is_policy_joint # joint policy
        utilities = zeros(size(uav.joint_actions, 2))
        # intruder signalling to take same sense
        mod_utilities = modify_utilities(uav, state, utilities)
        exp_utilities = zeros(length(uav.joint_actions[1, :]))
        for aidx in 1:size(uav.joint_actions, 2)
            if sign(uav.joint_actions[1, aidx]) == 1
                exp_utilities[aidx] = -Inf
            end
        end
    else
        utilities = zeros(length(uav.actions))
        # intruder signalling to take same sense
        mod_utilities = modify_utilities(uav, state, utilities)
        exp_utilities = zeros(length(uav.actions))
        for aidx in 1:length(uav.actions)
            if sign(uav.actions[aidx]) == 1
                exp_utilities[aidx] = -Inf
            end
        end
    end
    @test mod_utilities == exp_utilities
end

function test_select_action_and_signals_best()

    environment = build_environment()
    environment.uavs[1].action_selection_method = "best"
    environment.uavs[2].action_selection_method = "best"

    no_prev_signals = [:no_signal, :no_signal]

    uav = environment.uavs[1]
    if uav.is_policy_joint # joint policy
        utilities = zeros(size(uav.joint_actions, 2))
        utilities[1] = 1
        sense = :same_sense
        action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
        @test signals[1] == signals[2]
        @test action == -deg2rad(12)

        sense = :different_sense
        utilities = zeros(size(uav.joint_actions, 2))
        utilities[3] = 1
        action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
        @test signals[1] != signals[2]
        @test action == 0.0

        uav = environment.uavs[2]
        m = MersenneTwister(1)
        utilities = rand(m, size(uav.joint_actions, 2))
        sense = :neither_sense
        action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
        best_action_idx = indmax(utilities)
        best_action = uav.joint_actions[uav.uav_id, best_action_idx]
        @test action == best_action
        @test signals == [:no_signal, :no_signal]

        uav = environment.uavs[2]
        utilities = zeros(size(uav.joint_actions, 2))
        utilities[21] = 10
        sense = :same_sense
        action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
        @test action == deg2rad(6)
        @test signals[1] == signals[2]
        uav = environment.uavs[1]
        action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
        @test action == 0.0
        @test signals[1] == signals[2]

        utilities = collect(1.:36.)
        sense = :same_sense
        action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
        @test action == COC_ACTION
        @test signals[1] == signals[2]
        @test signals[1] == :no_signal
    else
        utilities = zeros(length(uav.actions))
        utilities[1] = 1
        sense = :same_sense
        action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
        @test signals[1] == signals[2]
        @test action == -deg2rad(12)

        sense = :different_sense
        utilities = zeros(length(uav.actions))
        utilities[3] = 1
        action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
        @test signals[1] != signals[2]
        @test action == 0.0

        uav = environment.uavs[2]
        m = MersenneTwister(1)
        utilities = rand(m, length(uav.actions))
        sense = :neither_sense
        action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
        best_action_idx = indmax(utilities)
        best_action = uav.actions[best_action_idx]
        @test action == best_action
        @test signals == [:no_signal, :no_signal]

        utilities = collect(1.:6.)
        sense = :same_sense
        action, signals = select_action_and_signals(uav, utilities, sense, no_prev_signals)
        @test action == COC_ACTION
        @test signals[1] == signals[2]
        @test signals[1] == :no_signal
    end
end

function test_select_action_and_signals_worst()
    environment = build_environment()
    environment.uavs[1].action_selection_method = "worst"
    environment.uavs[2].action_selection_method = "worst"
    uav = environment.uavs[1]
    
    if uav.is_policy_joint # joint policy
        utilities = ones(size(uav.joint_actions, 2)) * -1
        utilities[[6, 12, 18, 24, 30, 36]] = 1
        sense = :same_sense
        action, signals = select_action_and_signals(uav, utilities, sense, [:no_signal, :no_signal])
        @test action == COC_ACTION
        @test signals[1] == :no_signal
        @test signals[1] == signals[2]
    
        uav = environment.uavs[2]
        utilities[30:end] = 1
        action, signals = select_action_and_signals(uav, utilities, sense, [:no_signal, :no_signal])
        @test action == COC_ACTION
        @test signals[1] == :no_signal
        @test signals[1] == signals[2]
    
        uav = environment.uavs[1]
        utilities = ones(size(uav.joint_actions, 2)) * -1
        utilities[[1, 7, 13, 19, 25, 31]] = 1
        utilities[1] = -100
        action, signals = select_action_and_signals(uav, utilities, sense, [:no_signal, :no_signal])
        @test action != -deg2rad(12)
    
        utilities = ones(size(uav.joint_actions, 2))
        utilities[32:end] = -1
        action, signals = select_action_and_signals(uav, utilities, sense, [:no_signal, :no_signal])
        @test action == -deg2rad(12)
        @test signals[1] == :dont_turn_left
        @test signals[2] == :dont_turn_left
    
        uav = environment.uavs[1]
        utilities = ones(size(uav.joint_actions, 2))
        state = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
        state.signals = [:dont_turn_left, :dont_turn_left]
        utilities = modify_utilities(uav, state, utilities)
        action, signals = select_action_and_signals(uav, utilities, sense, state.signals)
        @test action == 0.0
        @test signals[1] == :dont_turn_left
        @test signals[2] == :dont_turn_left
    
        uav = environment.uavs[1]
        utilities = ones(size(uav.joint_actions, 2))
        state = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
        state.signals = [:dont_turn_right, :dont_turn_right]
        utilities = modify_utilities(uav, state, utilities)
        utilities[[3, 9, 15, 21, 27, 33]] = -1
        utilities[[4, 10, 16, 22, 28, 34]] = -1
        utilities[[6, 12, 18, 24, 30, 36]] = -1
        action, signals = select_action_and_signals(uav, utilities, sense, state.signals)
        @test action == deg2rad(12)
        @test signals[1] == :dont_turn_right
        @test signals[2] == :dont_turn_right
    
        uav = environment.uavs[1]
        utilities = ones(size(uav.joint_actions, 2))
        sense = :different_sense
        state = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
        state.signals = [:dont_turn_right, :dont_turn_right]
        utilities = modify_utilities(uav, state, utilities)
        utilities[[3, 9, 15, 21, 27, 33]] = -1
        utilities[[4, 10, 16, 22, 28, 34]] = -1
        utilities[[6, 12, 18, 24, 30, 36]] = -1
        action, signals = select_action_and_signals(uav, utilities, sense, state.signals)
        @test action == deg2rad(12)
        @test signals[1] == :dont_turn_right
        @test signals[2] == :dont_turn_left
    end
end

function test_select_action_and_signals_average()
    environment = build_environment()
    environment.uavs[1].action_selection_method = "average"
    environment.uavs[2].action_selection_method = "average"
    sense = :same_sense
    uav = environment.uavs[1]
    if uav.is_policy_joint # joint policy
        utilities = ones(size(uav.joint_actions, 2))
        state = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
        state.signals = [:dont_turn_right, :dont_turn_right]
        utilities = modify_utilities(uav, state, utilities)
        utilities[[3, 9, 15, 21, 27, 33]] = -1
        utilities[[4, 10, 16, 22, 28, 34]] = -.5
        utilities[[6, 12, 18, 24, 30, 36]] = -1
        utilities[17] = -10000
        action, signals = select_action_and_signals(uav, utilities, sense, state.signals)
        @test action == deg2rad(6)
        @test signals[1] == :dont_turn_right
        @test signals[2] == :dont_turn_right
    end
end

# non joint action tests
function test_select_equal_non_joint()
    environmentB = build_environment()
    environmentB.uavs[1].action_selection_method = "best"
    environmentB.uavs[2].action_selection_method = "best"

    environmentW = build_environment()
    environmentW.uavs[1].action_selection_method = "worst"
    environmentW.uavs[2].action_selection_method = "worst"

    environmentA = build_environment()
    environmentA.uavs[1].action_selection_method = "average"
    environmentA.uavs[2].action_selection_method = "average"

    no_prev_signals = [:no_signal, :no_signal]

    uavB = environmentB.uavs[1]
    uavW = environmentW.uavs[1]
    uavA = environmentA.uavs[1]

    # only a test for non-joint policies
    if !uavB.is_policy_joint # non joint policy
        println("i am in the correct test")
        utilitiesB = zeros(length(uavB.actions))
        utilitiesB[1] = 1
        senseB = :same_sense
        actionB, signalsB = select_action_and_signals(uavB, utilitiesB, senseB, no_prev_signals)
        @test signalsB[1] == signalsB[2]
        @test actionB == -deg2rad(12)
        
        utilitiesW = zeros(length(uavW.actions))
        utilitiesW[1] = 1
        senseW = :same_sense
        actionW, signalsW = select_action_and_signals(uavW, utilitiesW, senseW, no_prev_signals)
        @test signalsW[1] == signalsW[2]
        @test actionW == -deg2rad(12)

        utilitiesA = zeros(length(uavA.actions))
        utilitiesA[1] = 1
        senseA = :same_sense
        actionA, signalsA = select_action_and_signals(uavA, utilitiesA, senseA, no_prev_signals)
        @test signalsA[1] == signalsA[2]
        @test actionA == -deg2rad(12)

        @test actionB == actionW == actionA
        @test signalsB == signalsW == signalsA

        senseB = :different_sense
        utilitiesB = zeros(length(uavB.actions))
        utilitiesB[3] = 1
        actionB, signalsB = select_action_and_signals(uavB, utilitiesB, senseB, no_prev_signals)
        @test signalsB[1] != signalsB[2]
        @test actionB == 0.0

        senseW = :different_sense
        utilitiesW = zeros(length(uavW.actions))
        utilitiesW[3] = 1
        actionW, signalsW = select_action_and_signals(uavW, utilitiesW, senseW, no_prev_signals)
        @test signalsW[1] != signalsW[2]
        @test actionW == 0.0

        senseA = :different_sense
        utilitiesA = zeros(length(uavA.actions))
        utilitiesA[3] = 1
        actionA, signalsA = select_action_and_signals(uavA, utilitiesA, senseA, no_prev_signals)
        @test signalsA[1] != signalsA[2]
        @test actionA == 0.0

        @test actionB == actionW == actionA
        @test signalsB == signalsW == signalsA

        uavB = environmentB.uavs[2]
        m = MersenneTwister(1)
        utilitiesB = rand(m, length(uavB.actions))
        senseB = :neither_sense
        actionB, signalsB = select_action_and_signals(uavB, utilitiesB, senseB, no_prev_signals)
        best_action_idxB = indmax(utilitiesB)
        best_actionB = uavB.actions[best_action_idxB]
        @test actionB == best_actionB
        @test signalsB == [:no_signal, :no_signal]

        uavW = environmentW.uavs[2]
        m = MersenneTwister(1)
        utilitiesW = rand(m, length(uavW.actions))
        senseW = :neither_sense
        actionW, signalsW = select_action_and_signals(uavW, utilitiesW, senseW, no_prev_signals)
        best_action_idxW = indmax(utilitiesW)
        best_actionW = uavW.actions[best_action_idxW]
        @test actionW == best_actionW
        @test signalsW == [:no_signal, :no_signal]

        uavA = environmentA.uavs[2]
        m = MersenneTwister(1)
        utilitiesA = rand(m, length(uavA.actions))
        senseA = :neither_sense
        actionA, signalsA = select_action_and_signals(uavA, utilitiesA, senseA, no_prev_signals)
        best_action_idxA = indmax(utilitiesA)
        best_actionA = uavA.actions[best_action_idxA]
        @test actionA == best_actionA
        @test signalsA == [:no_signal, :no_signal]

        @test actionB == actionW == actionA
        @test signalsB == signalsW == signalsA

        utilitiesB = collect(1.:6.)
        senseB = :same_sense
        actionB, signalsB = select_action_and_signals(uavB, utilitiesB, senseB, no_prev_signals)
        @test actionB == COC_ACTION
        @test signalsB[1] == signalsB[2]
        @test signalsB[1] == :no_signal

        utilitiesW = collect(1.:6.)
        senseW = :same_sense
        actionW, signalsW = select_action_and_signals(uavW, utilitiesW, senseW, no_prev_signals)
        @test actionW == COC_ACTION
        @test signalsW[1] == signalsW[2]
        @test signalsW[1] == :no_signal

        utilitiesA = collect(1.:6.)
        senseA = :same_sense
        actionA, signalsA = select_action_and_signals(uavA, utilitiesA, senseA, no_prev_signals)
        @test actionA == COC_ACTION
        @test signalsA[1] == signalsA[2]
        @test signalsA[1] == :no_signal

        @test actionB == actionW == actionA
        @test signalsB == signalsW == signalsA 
    end

    uavB = environmentB.uavs[1]
    uavW = environmentW.uavs[1]
    uavA = environmentA.uavs[1]
    
    if !uavB.is_policy_joint # non joint policy
        println("i am in the correct test")
        
        utilitiesB = ones(length(uavB.actions)) * -1
        utilitiesB[6] = 1
        senseB = :same_sense
        actionB, signalsB = select_action_and_signals(uavB, utilitiesB, senseB, [:no_signal, :no_signal])
        @test actionB == COC_ACTION
        @test signalsB[1] == :no_signal
        @test signalsB[1] == signalsB[2]

        utilitiesW = ones(length(uavW.actions)) * -1
        utilitiesW[6] = 1
        senseW = :same_sense
        actionW, signalsW = select_action_and_signals(uavW, utilitiesW, senseW, [:no_signal, :no_signal])
        @test actionW == COC_ACTION
        @test signalsW[1] == :no_signal
        @test signalsW[1] == signalsW[2]

        utilitiesA = ones(length(uavA.actions)) * -1
        utilitiesA[6] = 1
        senseA = :same_sense
        actionA, signalsB = select_action_and_signals(uavA, utilitiesA, senseA, [:no_signal, :no_signal])
        @test actionA == COC_ACTION
        @test signalsA[1] == :no_signal
        @test signalsA[1] == signalsA[2]

        @test actionB == actionW == actionA
        @test signalsB == signalsW == signalsA 
    
        uavB = environmentB.uavs[2]
        utilitiesB[end] = 1
        actionB, signalsB = select_action_and_signals(uavB, utilitiesB, senseB, [:no_signal, :no_signal])
        @test actionB == COC_ACTION
        @test signalsB[1] == :no_signal
        @test signalsB[1] == signalsB[2]

        uavW = environmentW.uavs[2]
        utilitiesW[end] = 1
        actionW, signalsW = select_action_and_signals(uavW, utilitiesW, senseW, [:no_signal, :no_signal])
        @test actionW == COC_ACTION
        @test signalsW[1] == :no_signal
        @test signalsW[1] == signalsW[2]

        uavA = environmentA.uavs[2]
        utilitiesA[end] = 1
        actionA, signalsA = select_action_and_signals(uavA, utilitiesA, senseA, [:no_signal, :no_signal])
        @test actionA == COC_ACTION
        @test signalsA[1] == :no_signal
        @test signalsA[1] == signalsA[2]

        @test actionB == actionW == actionA
        @test signalsB == signalsW == signalsA
    
        uavB = environmentB.uavs[1]
        utilitiesB = ones(length(uavB.actions)) * -1
        utilitiesB[1] = -100
        actionB, signalsB = select_action_and_signals(uavB, utilitiesB, senseB, [:no_signal, :no_signal])
        @test actionB != -deg2rad(12)

        uavW = environmentW.uavs[1]
        utilitiesW = ones(length(uavW.actions)) * -1
        utilitiesW[1] = -100
        actionW, signalsW = select_action_and_signals(uavW, utilitiesW, senseW, [:no_signal, :no_signal])
        @test actionW != -deg2rad(12)

        uavA = environmentA.uavs[1]
        utilitiesA = ones(length(uavA.actions)) * -1
        utilitiesA[1] = -100
        actionA, signalsA = select_action_and_signals(uavA, utilitiesA, senseA, [:no_signal, :no_signal])
        @test actionA != -deg2rad(12)

        @test actionB == actionW == actionA
        @test signalsB == signalsW == signalsA

        uavB = environmentB.uavs[1]
        utilitiesB = ones(length(uavB.actions))
        stateB = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
        stateB.signals = [:dont_turn_left, :dont_turn_left]
        utilitiesB = modify_utilities(uavB, stateB, utilitiesB)
        actionB, signalsB = select_action_and_signals(uavB, utilitiesB, senseB, stateB.signals)
        @test actionB == 0.0
        @test signalsB[1] == :dont_turn_left
        @test signalsB[2] == :dont_turn_left

        uavW = environmentW.uavs[1]
        utilitiesW = ones(length(uavW.actions))
        stateW = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
        stateW.signals = [:dont_turn_left, :dont_turn_left]
        utilitiesW = modify_utilities(uavW, stateW, utilitiesW)
        actionW, signalsW = select_action_and_signals(uavW, utilitiesW, senseW, stateW.signals)
        @test actionW == 0.0
        @test signalsW[1] == :dont_turn_left
        @test signalsW[2] == :dont_turn_left

        uavA = environmentA.uavs[1]
        utilitiesA = ones(length(uavA.actions))
        stateA = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
        stateA.signals = [:dont_turn_left, :dont_turn_left]
        utilitiesA = modify_utilities(uavA, stateA, utilitiesA)
        actionA, signalsA = select_action_and_signals(uavA, utilitiesA, senseA, stateA.signals)
        @test actionA == 0.0
        @test signalsA[1] == :dont_turn_left
        @test signalsA[2] == :dont_turn_left

        @test actionB == actionW == actionA
        @test signalsB == signalsW == signalsA

        uavB = environmentB.uavs[1]
        utilitiesB = ones(length(uavB.actions))
        stateB = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
        stateB.signals = [:dont_turn_right, :dont_turn_right]
        utilitiesB = modify_utilities(uavB, stateB, utilitiesB)
        utilitiesB[3] = -1
        utilitiesB[4] = -1
        utilitiesB[6] = -1
        actionB, signalsB = select_action_and_signals(uavB, utilitiesB, senseB, stateB.signals)
        @test actionB == deg2rad(12)
        @test signalsB[1] == :dont_turn_right
        @test signalsB[2] == :dont_turn_right

        uavW = environmentW.uavs[1]
        utilitiesW = ones(length(uavW.actions))
        stateW = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
        stateW.signals = [:dont_turn_right, :dont_turn_right]
        utilitiesW = modify_utilities(uavW, stateW, utilitiesW)
        utilitiesW[3] = -1
        utilitiesW[4] = -1
        utilitiesW[6] = -1
        actionW, signalsW = select_action_and_signals(uavW, utilitiesW, senseW, stateW.signals)
        @test actionW == deg2rad(12)
        @test signalsW[1] == :dont_turn_right
        @test signalsW[2] == :dont_turn_right

        uavA = environmentA.uavs[1]
        utilitiesA = ones(length(uavA.actions))
        stateA = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
        stateA.signals = [:dont_turn_right, :dont_turn_right]
        utilitiesA = modify_utilities(uavA, stateA, utilitiesA)
        utilitiesA[3] = -1
        utilitiesA[4] = -1
        utilitiesA[6] = -1
        actionA, signalsA = select_action_and_signals(uavA, utilitiesA, senseA, stateA.signals)
        @test actionA == deg2rad(12)
        @test signalsA[1] == :dont_turn_right
        @test signalsA[2] == :dont_turn_right

        @test actionB == actionW == actionA
        @test signalsB == signalsW == signalsA
    end

    uavB = environmentB.uavs[1]
    uavW = environmentW.uavs[1]
    uavA = environmentA.uavs[1]
    sense = :same_sense
    
    if !uavB.is_policy_joint # non joint policy
        println("i am in the correct test")

        utilitiesB = ones(length(uavB.actions))
        stateB = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
        stateB.signals = [:dont_turn_right, :dont_turn_right]
        utilitiesB = modify_utilities(uavB, stateB, utilitiesB)
        utilitiesB[3] = -1
        utilitiesB[4] = -.5
        utilitiesB[6] = -1
        utilitiesB[5] = -10000
        actionB, signalsB = select_action_and_signals(uavB, utilitiesB, senseB, stateB.signals)
        @test actionB == deg2rad(6)
        @test signalsB[1] == :dont_turn_right
        @test signalsB[2] == :dont_turn_right

        utilitiesW = ones(length(uavW.actions))
        stateW = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
        stateW.signals = [:dont_turn_right, :dont_turn_right]
        utilitiesW = modify_utilities(uavW, stateW, utilitiesW)
        utilitiesW[3] = -1
        utilitiesW[4] = -.5
        utilitiesW[6] = -1
        utilitiesW[5] = -10000
        actionW, signalsW = select_action_and_signals(uavW, utilitiesW, senseW, stateW.signals)
        @test actionW == deg2rad(6)
        @test signalsW[1] == :dont_turn_right
        @test signalsW[2] == :dont_turn_right

        utilitiesA = ones(length(uavA.actions))
        stateA = State([UAVState(ones(Float64, 8)...), UAVState(ones(Float64, 8)...)])
        stateA.signals = [:dont_turn_right, :dont_turn_right]
        utilitiesA = modify_utilities(uavA, stateA, utilitiesA)
        utilitiesA[3] = -1
        utilitiesA[4] = -.5
        utilitiesA[6] = -1
        utilitiesA[5] = -10000
        actionA, signalsA = select_action_and_signals(uavA, utilitiesA, senseA, stateA.signals)
        @test actionA == deg2rad(6)
        @test signalsA[1] == :dont_turn_right
        @test signalsA[2] == :dont_turn_right
        
        @test actionB == actionW == actionA
        @test signalsB == signalsW == signalsA
    end
end


function main()
    test_modify_utilities()
    test_select_action_and_signals_best()
    test_select_action_and_signals_worst()
    test_select_action_and_signals_average()
    test_select_equal_non_joint()
end

@time main()
