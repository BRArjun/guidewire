# Locust Load Testing Scenarios

## Overview
This repository contains various Locust load testing scenarios for evaluating web application performance under different conditions. Each scenario simulates user behavior and traffic patterns using predefined request sequences.

## Load Testing Scenarios

### 1. **Step Load Test**
- Gradually increases the number of users in steps.
- Users are spawned at a defined rate until a maximum limit is reached.
- After a predefined time limit, the test resets and repeats.

### 2. **Cyclic Load with Gaps**
- Simulates fluctuating traffic with periodic gaps.
- Alternates between active user load and periods with minimal traffic.
- Useful for testing applications that experience periodic downtime or off-peak usage.

### 3. **Cyclic Load with Jumps Up**
- Introduces sudden spikes in user traffic.
- After regular cycles, the user load triples at predefined intervals.
- Useful for testing resilience against sudden traffic surges.

### Configuring Load Test Parameters
Modify parameters such as `max_users`, `spawn_rate`, and cycle intervals directly in the test scripts to adjust the load behavior.
