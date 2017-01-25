#=
Create a single coordination table from multiple policies.
The multiple policies are from the optimal policy sweep. 
This will only work on JOINT policy tables.
=#

# for access to constant values
include("build_simulation.jl")
include("tables.jl")

# setup the filepaths for the various policies
policy_1_filepath = "../../data/qvalue_tables/j1.jld"
policy_2_filepath = "../../data/qvalue_tables/j2.jld"
policy_3_filepath = "../../data/qvalue_tables/j3.jld"
policy_4_filepath = "../../data/qvalue_tables/j4.jld"
policy_5_filepath = "../../data/qvalue_tables/j5.jld"
policy_6_filepath = "../../data/qvalue_tables/j6.jld"
policy_7_filepath = "../../data/qvalue_tables/j7.jld"
policy_8_filepath = "../../data/qvalue_tables/j8.jld"
policy_9_filepath = "../../data/qvalue_tables/j9.jld"
policy_10_filepath = "../../data/qvalue_tables/j10.jld"
policy_11_filepath = "../../data/qvalue_tables/j11.jld"
policy_12_filepath = "../../data/qvalue_tables/j12.jld"
policy_13_filepath = "../../data/qvalue_tables/j13.jld"
policy_14_filepath = "../../data/qvalue_tables/j14.jld"
policy_15_filepath = "../../data/qvalue_tables/j15.jld"
println("all filepaths declared")

policy_name = "solQ"

# load the various q_tables -> all have = dimensions 
q_table_1 = load(policy_1_filepath, policy_name)
q_table_2 = load(policy_2_filepath, policy_name)
q_table_3 = load(policy_3_filepath, policy_name)
q_table_4 = load(policy_4_filepath, policy_name)
q_table_5 = load(policy_5_filepath, policy_name)
q_table_6 = load(policy_6_filepath, policy_name)
q_table_7 = load(policy_7_filepath, policy_name)
q_table_8 = load(policy_8_filepath, policy_name)
q_table_9 = load(policy_9_filepath, policy_name)
q_table_10 = load(policy_10_filepath, policy_name)
q_table_11 = load(policy_11_filepath, policy_name)
q_table_12 = load(policy_12_filepath, policy_name)
q_table_13 = load(policy_13_filepath, policy_name)
q_table_14 = load(policy_14_filepath, policy_name)
q_table_15 = load(policy_15_filepath, policy_name)
println("all qtables loaded")

# set the grid and number of states
q_grid = RectangleGrid(RANGES, THETAS, BEARINGS, SPEEDS, SPEEDS, RESPONDINGS, RESPONDINGS)
num_q_table_states = reduce(*, q_grid.cut_counts) + 1

# check the table sizes with number of states
msg = "the q_table (rows = $(size(q_table_1, 1))) does not match the 
       q_grid (length = $num_q_table_states), which was created using constant values"
       @assert num_q_table_states == size(q_table_1, 1) msg
       @assert num_q_table_states == size(q_table_2, 1) msg
       @assert num_q_table_states == size(q_table_3, 1) msg
       @assert num_q_table_states == size(q_table_4, 1) msg
       @assert num_q_table_states == size(q_table_5, 1) msg
       @assert num_q_table_states == size(q_table_6, 1) msg
       @assert num_q_table_states == size(q_table_7, 1) msg
       @assert num_q_table_states == size(q_table_8, 1) msg
       @assert num_q_table_states == size(q_table_9, 1) msg
       @assert num_q_table_states == size(q_table_10, 1) msg
       @assert num_q_table_states == size(q_table_11, 1) msg
       @assert num_q_table_states == size(q_table_12, 1) msg
       @assert num_q_table_states == size(q_table_13, 1) msg
       @assert num_q_table_states == size(q_table_14, 1) msg
       @assert num_q_table_states == size(q_table_15, 1) msg
println("the size of all q_tables checked")

# find the joint actions that are useless in finding a coordination table
# straight and COC
valid = Int64[]
joint_actions = get_joint_actions(ACTIONS)
# make it so that we do not select joint actions 
# with straight or COC as one of the options so remove 
# those from the options
for aidx in 1:size(joint_actions,2)
	if  joint_actions[1, aidx] == 0. || joint_actions[2, aidx] == 0. || 
           	joint_actions[1, aidx] == COC_ACTION || joint_actions[2, aidx] == COC_ACTION
    else
    	push!(valid,aidx)
    end
end

# update qtables and joint actions to reflect the useful ones
q_table_1 = q_table_1[:,collect(valid)]
q_table_2 = q_table_2[:,collect(valid)]
q_table_3 = q_table_3[:,collect(valid)]
q_table_4 = q_table_4[:,collect(valid)]
q_table_5 = q_table_5[:,collect(valid)]
q_table_6 = q_table_6[:,collect(valid)]
q_table_7 = q_table_7[:,collect(valid)]
q_table_8 = q_table_8[:,collect(valid)]
q_table_9 = q_table_9[:,collect(valid)]
q_table_10 = q_table_10[:,collect(valid)]
q_table_11 = q_table_11[:,collect(valid)]
q_table_12 = q_table_12[:,collect(valid)]
q_table_13 = q_table_13[:,collect(valid)]
q_table_14 = q_table_14[:,collect(valid)]
q_table_15 = q_table_15[:,collect(valid)]
joint_actions = joint_actions[:,collect(valid)]
println("sized down joint actions and qvalues")

# we want to adjust the q_tables so they are comparable -> z-score
q_table_1 = (q_table_1-mean(q_table_1))/std(q_table_1)
q_table_2 = (q_table_2-mean(q_table_2))/std(q_table_2)
q_table_3 = (q_table_3-mean(q_table_3))/std(q_table_3)
q_table_4 = (q_table_4-mean(q_table_4))/std(q_table_4)
q_table_5 = (q_table_5-mean(q_table_5))/std(q_table_5)
q_table_6 = (q_table_6-mean(q_table_6))/std(q_table_6)
q_table_7 = (q_table_7-mean(q_table_7))/std(q_table_7)
q_table_8 = (q_table_8-mean(q_table_8))/std(q_table_8)
q_table_9 = (q_table_9-mean(q_table_9))/std(q_table_9)
q_table_10 = (q_table_10-mean(q_table_10))/std(q_table_10)
q_table_11 = (q_table_11-mean(q_table_11))/std(q_table_11)
q_table_12 = (q_table_12-mean(q_table_12))/std(q_table_12)
q_table_13 = (q_table_13-mean(q_table_13))/std(q_table_13)
q_table_14 = (q_table_14-mean(q_table_14))/std(q_table_14)
q_table_15 = (q_table_15-mean(q_table_15))/std(q_table_15)
println("adjusted all qtable values")

# allocate coordination table
c_table_best = zeros(size(q_table_1, 1), 1)
c_table_average = zeros(size(q_table_1, 1), 1)
c_table_worst = zeros(size(q_table_1, 1), 1)
println("coordination tables allocated")

println("making the tables...")
# go through each row setting the value
for ridx in 1:size(q_table_1, 1)
    utilities_1 = q_table_1[ridx, :]
    utilities_2 = q_table_2[ridx, :]
    utilities_3 = q_table_3[ridx, :]
    utilities_4 = q_table_4[ridx, :]
    utilities_5 = q_table_5[ridx, :]
    utilities_6 = q_table_6[ridx, :]
    utilities_7 = q_table_7[ridx, :]
    utilities_8 = q_table_8[ridx, :]
    utilities_9 = q_table_9[ridx, :]
    utilities_10 = q_table_10[ridx, :]
    utilities_11 = q_table_11[ridx, :]
    utilities_12 = q_table_12[ridx, :]
    utilities_13 = q_table_13[ridx, :]
    utilities_14 = q_table_14[ridx, :]
    utilities_15 = q_table_15[ridx, :]
    utilities = [utilities_1; utilities_2; utilities_3; utilities_4; utilities_5; 
    	         utilities_6; utilities_7; utilities_8; utilities_9; utilities_10; 
    	         utilities_11; utilities_12; utilities_13; utilities_14; utilities_15]

    ### BEST CASE COORDINATION TABLE
    # from all the coordination tables, find the action with the largest utility
    policy_max_idx, action_max_idx = ind2sub(size(utilities),indmax(utilities))
    actions_best = joint_actions[:, action_max_idx]

    # if the signs of the two actions match then 
    # signal 1 for requiring same sense actions
    # note that we removed COC and straight above so 
    # does not come up here therefore case SAME includes:
    # right + right 
    # left + left
    if sign(actions_best[1]) == sign(actions_best[2])
        c_table_value_best = 1

    # otherwise, signal to only do opposite sense
    # therefore case DIFF includes:
    # right + left
    # left + right
    else
        c_table_value_best = -1
    end
	c_table_best[ridx] = c_table_value_best

    ### WORST CASE COORDINATION TABLE
    # for each coordination table find the minimum utility action,
    # then select the action from the minimum actions that has the 
    # maximum utility
    action_min_idx = indmax(minimum(utilities,1))
    actions_worst = joint_actions[:, action_min_idx]

    # if the signs of the two actions match then 
    # signal 1 for requiring same sense actions
    # note that we removed COC and straight above so 
    # does not come up here therefore case SAME includes:
    # right + right 
    # left + left
    if sign(actions_worst[1]) == sign(actions_worst[2])
        c_table_value_worst = 1

    # otherwise, signal to only do opposite sense
    # therefore case DIFF includes:
    # right + left
    # left + right
    else
        c_table_value_worst = -1
    end
    c_table_worst[ridx] = c_table_value_worst

    ### AVERAGE CASE COORDINATION TABLE
    # for each action, find the average utility of that action across 
    # all of the coordination tables, then select the action with the 
    # highest average utility
    action_avg_idx = indmax(mean(utilities,1))
    actions_average = joint_actions[:, action_avg_idx]

    # if the signs of the two actions match then 
    # signal 1 for requiring same sense actions
    # note that we removed COC and straight above so 
    # does not come up here therefore case SAME includes:
    # right + right 
    # left + left
    if sign(actions_average[1]) == sign(actions_average[2])
        c_table_value_average = 1

    # otherwise, signal to only do opposite sense
    # therefore case DIFF includes:
    # right + left
    # left + right
    else
        c_table_value_average = -1
    end
    c_table_average[ridx] = c_table_value_average
end
println("tables made!")
println("table sizes are as follows:")
println("best:")
println(size(c_table_best))
println("worst:")
println(size(c_table_worst))
println("average:")
println(size(c_table_average))

println("does best equal worst?")
println(c_table_best == c_table_worst)
println("does best equal average?")
println(c_table_best == c_table_average)
println("does average equal worst?")
println(c_table_average == c_table_worst)

println("saving the tables...")
save("../../data/c_tables/ct_best.jld", "solQ", c_table_best)
save("../../data/c_tables/ct_worst.jld", "solQ", c_table_worst)
save("../../data/c_tables/ct_average.jld", "solQ", c_table_average)
println("tables saved!")