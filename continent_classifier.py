from sklearn import tree
#Features: Popuplation denisty(pop./mi^2),total population, Area(mi^2)

#UK,Germany,Serbia,Sweden,Spain #USA,Cuba,Nicaragua,Honduras,Dominican Republic #Brazil,Venezuela,Ecuador,Argentina,Chile
#South Africa,Nigeria,Sudan,Algeria,DR Congo #China,Malaysia,Afghanistan,India,Japan
features = [[694,65110000,93788],[601,82800000,137903],[236,7076372,29913],[58,10046418,173860],[238,46438422,195364],
[264,11239004,42426],[86,325295437,3796742],[135,6262703,46884],[205,8866351,43433],[570,10528000,18485],
[83,28946101,353841],[62,202768562,3287956],[167,16529700,98686],[36,40117096,1073518],[57,16634603,291930],
[111,52981991,471359],[539,191836000,356669],[44,30894000,710251],[41,38700000,919595],[94,85026000,905446],
[372,1383909395,3722342],[251,32110700,127724],[101,25500100,249347],[1038,1317655596,1269211],[868,126730000,145925]]
labels = ["Europe","Europe","Europe","Europe","Europe",
"North America","North America","North America","North America","North America",
"South America","South America","South America","South America","South America",
"Africa","Africa","Africa","Africa","Africa",
"Asia","Asia","Asia","Asia","Asia"]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)

print clf.predict([[163,123518000,759516]])#Mexico
print clf.predict([[319,67032000,210026]])#France
print clf.predict([[725,92700000,127882]])#Vietnam
print clf.predict([[62,30475144,1496225]])#Peru
print clf.predict([[31,14528662,482077]])#Mali
print clf.predict([[272,8767919,32386]])#Austria
print clf.predict([[119,3405813,28640]])#Panama
print clf.predict([[23,10389913,424164]])#Colombia
print clf.predict([[249,101853000,410678]])#Ethiopia
print clf.predict([[900,104214282,115831]])#Thailand