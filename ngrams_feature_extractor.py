import sklearn


def make_train_pair(filename):
    h5 = open_h5_file_read(filename)
    title = get_title(h5)
    pitches = get_segments_pitches(h5)[:11] # limit: only look at beginning
    pitch_diffs = [pitches[i] - pitches[i - 1] for i in xrange(1, len(pitches))]
    h5.close()
    return {'title': title, 'pitch_diffs': pitch_diffs}
# some lines omitted
neigh = NearestNeighbors(n_neighbors=1) # predict the closest song
# a title list is also maintained
neigh.fit([sum(diff) for diff in pitch_diff_list[5000:]])
neigh.kneighbors(sum(pitch_diff_list[2029])) # example prediction