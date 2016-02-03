from sgflib.sgflib import SGFParser

with open('../SGF/friday_tournament.sgf','r') as game_file:
    sgf_object = SGFParser(game_file.read())
    c = sgf_object.parse().cursor()
    while(True): print c.next()
