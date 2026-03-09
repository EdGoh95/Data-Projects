#!/usr/bin/env python3
"""
Machine Learning Engineering In Action (Manning Publication)
Chapter 14 - Writing production code
"""
import numpy as np

#%% Feature Monitoring
original_data = np.array(
    [199.9, 207.2, 185.3, 312.7, 277.3, 373.5, 456.2, 121.9, 155.3, 167.4, 222.1, 139.2, 157.8,
     212.1, 241.8, 277.2, 145.6, 192.2, 203.4, 166.5, 177.3, 193.1, 134.9, 199.9, 217.4])
boundary_data = np.array([155.1, 173.2, 183.9, 217.3, 224.9])

prior_to_shift = np.append(original_data, boundary_data)
prior_stats = {}
prior_stats['Prior Standard Deviation'] = np.std(prior_to_shift)
prior_stats['Prior Mean'] = np.mean(prior_to_shift)
prior_stats['Prior Median'] = np.median(prior_to_shift)
prior_stats['Prior Minimum'] = np.min(prior_to_shift)
prior_stats['Prior Maximum'] = np.max(prior_to_shift)

post_shift = np.append(boundary_data, np.full(original_data.size, 0))
post_stats = {}
post_stats['Post Standard Deviation'] = np.std(post_shift)
post_stats['Post Mean'] = np.mean(post_shift)
post_stats['Post Median'] = np.median(post_shift)
post_stats['Post Minimum'] = np.min(post_shift)
post_stats['Post Maximum'] = np.max(post_shift)

error_message = 'Something seems amiss in our sales data! '
if post_stats['Post Mean'] <= prior_stats['Prior Minimum']:
    print(error_message + 'Current mean is lower than the minimum value of the training data!')
if post_stats['Post Mean'] >= prior_stats['Prior Maximum']:
    print(error_message + 'Current mean is greater than the maximum value of the training data!')
if ~(prior_stats['Prior Standard Deviation']/2 <= post_stats['Post Standard Deviation']
     < 2 * prior_stats['Prior Standard Deviation']):
    print(error_message + 'Current standard deviation is way out of bounds!')