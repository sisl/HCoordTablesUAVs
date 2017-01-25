#=
This file defines the uav type, which uses different strategies to 
make decisions based on the current state of the system.
=#

# type to hold the actions and signals emitted from uavs
type UAVActions
    actions::Vector{Float64}
    # signals sent by the uavs to coordinate next actions
    signals::Vector{Symbol}
    function UAVActions(num_uavs::Int64)
        actions = Array(Float64, num_uavs)
        signals = [:no_signal, :no_signal]
        return new(actions, signals)
    end
end

# type for uav 
type UAV
    # id gives index into tables for this uav 
    uav_id::Int64
    # actions
    actions::Vector{Float64}
    num_actions::Int64
    joint_actions::Matrix{Float64}
    num_joint_actions::Int64
    # tables
    policy_table::PolicyTable
    coordination_table::CoordinationTable
    # type of policy
    is_policy_joint::Bool
    # strategy
    strategy::Function
    action_selection_method::ASCIIString
    # sensor accuracy information
    # we now need to pass the random number generator to the uav:
    rng::MersenneTwister
    # gaussian sensors, mean = 0, std dev these values
    sensor_range_std::Float64
    sensor_theta_std::Float64
    sensor_psi_std::Float64
    sensor_velocity_std::Float64
    use_sensor_noise::Bool
    function UAV(uav_id::Int64, actions::Vector{Float64}, joint_actions::Matrix{Float64},
            policy_table::PolicyTable, coordination_table::CoordinationTable, is_policy_joint::Bool,
            strategy::Function, action_selection_method::ASCIIString, rng::MersenneTwister, 
            sensor_range_std::Float64, sensor_theta_std::Float64, sensor_psi_std::Float64, 
            sensor_velocity_std::Float64, use_sensor_noise::Bool)
        return new(uav_id, actions, length(actions), joint_actions, size(joint_actions, 2),
                policy_table, coordination_table, is_policy_joint, strategy, action_selection_method, 
                rng, sensor_range_std, sensor_theta_std, sensor_psi_std, sensor_velocity_std, use_sensor_noise)
    end
end

#=
Description:
Delegates to the strategy of the uav to decide which action to take

Parameters:
- uav: the uav deciding which action to take
- state: state object which contains all variable information
    needed by the uav to decide which action to take

Return Value:
- action: float value that is the action of this uav
- signal: float value indiciating coordination action
- utilities: the utilities used to derive the actions
- belief_state: the belief_state used to index the policy table
- different_action: whether or not the coordination table prevented
    the taking of a different action from the greedy case and 
    the coordinationt table action was coc. This is used to 
    wake the pilot.
=#
function get_action(uav::UAV, state::State)
    
    # compute belief state 
    belief_state = get_belief_state(uav, state)

    # get the baseline utilities from the policy table
    utilities = get_utilities(uav.policy_table, belief_state)

    # delegate to the strategy to decide which action to take
    # as well as whether to signal the other plane
    action, signals, different_action = uav.strategy(uav, state, utilities)

    return action, signals, utilities, belief_state, different_action
end

#= 
Description:
Converts the state from the master point of view to the slave point of view
to select the correct action.
=#
function belief_state_from_slave(belief_state::Array{Float64,1})
    v0 = belief_state[4]
    v1 = belief_state[5]
    slave_belief_state = belief_state
    # [1] stays the same
    # [2] changes - requires calc
    # [3] changes - requires calc
    # [4] becomes [5] - no calculation
    slave_belief_state[4] = v1
    # [5] becomes [4] - no calculation
    slave_belief_state[5] = v0
    # [6] stays the same

    # solve for x and y coordinates
    range = belief_state[1]
    theta = belief_state[2] # rad
    psi = belief_state[3] # rad
    p1 = [0;0]
    p2 = [range*cos(theta);range*sin(theta)]

    # move the point 1 and rotate
    p1 = p1-p2
    # rotate point 1
    rot = [cos(psi) sin(psi);-sin(psi) cos(psi)]
    p1 = rot*p1

    # calculate new theta
    thetaNew = atan2(p1[2],p1[1]) # rad
    while thetaNew > pi
        thetaNew = thetaNew - 2*pi 
    end
    while thetaNew < -pi
        thetaNew = thetaNew + 2*pi
    end

    # calculate new psi
    psiNew = 2*pi-psi # rad
    # ensure new psi is between 0 and 2*pi
    while psiNew < 0
        psiNew = psiNew+2*pi
    end

    slave_belief_state[2] = thetaNew # rad
    slave_belief_state[3] = psiNew # rad

    return slave_belief_state
end

#=
Description:
Converts the state to polar format and check that it is in bounds of the policy

Parameters:
- uav: the uav 
- state: state object to convert to polar

Return Value:
- polar_state: the state converted to polar coordinates, which consists of 
    [xr, yr, pr, vown, vint, resp, resp] to [rho, theta, psi, vown, vint, resp, resp]
=#
function get_polar_state(uav::UAV, state::State)
    # delegate to state for conversion
    if uav.is_policy_joint # joint policy
        polar_state = to_polar(state)
    else
        polar_state = to_polar_non_joint(state)
    end

    # get bounds of grid
    max_r = maximum(uav.policy_table.grid.cutPoints[1])
    min_t = minimum(uav.policy_table.grid.cutPoints[2])
    max_t = maximum(uav.policy_table.grid.cutPoints[2])

    # check that the new polar state is inside of the bounds
    # and if not return the terminal state
    r = polar_state[1]
    theta = polar_state[2]
    if r > max_r || theta < min_t || theta > max_t
        polar_state = TERMINAL_POLAR_STATE
    end

    return polar_state
end

#=
Description:
Populates a belief state of the uav

Parameters:
- uav: the uav, the belief state of which is being calculated
- state: the state converted to belief state

Return Value:
- belief_state: probability distribution over possible states
=#
function get_belief_state(uav::UAV, state::State)
    # initialize belief as sparse vector over all possible states
    belief_state = spzeros(uav.policy_table.num_states, 1)

    # convert state to polar
    polar_state = get_polar_state(uav, state)

    if !uav.is_policy_joint 
        if uav.uav_id != 1
            if polar_state != TERMINAL_POLAR_STATE
                polar_state = belief_state_from_slave(polar_state)
            end
        end
    end

    # add sensor error if using noisy sensors
    if uav.use_sensor_noise
        sensor_error_state = apply_sensor_error(uav, polar_state)
    else
        sensor_error_state = polar_state
    end

    # populate belief_state
    if polar_state == TERMINAL_POLAR_STATE
        belief_state[end] = 1.0
    else        
        indices, weights = interpolants(uav.policy_table.grid, sensor_error_state)
        belief_state[indices] = weights
    end

    return belief_state
end

#=
Description:
Returns the id value of the other uav 

Parameters:
- uav: the current uav

Return Value:
- id: int value of the other uavs id
=#
function get_other_uav_id(uav::UAV)
    if uav.uav_id == 1
        return 2
    else
        return 1
    end
end

function apply_sensor_error(uav::UAV, polar_state::Vector{Float64})
    # copy polar state because it might be used 
    # elsewhere in the above function and it's not
    # obvious whether we might want to use the 
    # original or a noisy version above
    sensor_error_state = deepcopy(polar_state)

    # range
    sensor_error_state[1] = (sensor_error_state[1]  
        + randn(uav.rng) * uav.sensor_range_std)
    sensor_error_state[2] = (sensor_error_state[2]  
        + randn(uav.rng) * uav.sensor_theta_std)
    sensor_error_state[3] = (sensor_error_state[3]  
        + randn(uav.rng) * uav.sensor_psi_std)

    # apply noise to other planes velocity
    if uav.uav_id == 1
        other_uav_velocity_idx = 5
    else
        other_uav_velocity_idx = 4
    end
    sensor_error_state[other_uav_velocity_idx] = (
        sensor_error_state[other_uav_velocity_idx]  
        + randn(uav.rng) * uav.sensor_velocity_std)

    return sensor_error_state
end