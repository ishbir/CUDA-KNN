import subprocess
import re

def avg_col(table, col):
    avg = 0.0
    for row in table:
        avg += float(row[col])
    avg = avg/len(table)
    return ("%." + str(len(table[0][col].split('.')[1])) + "f") % avg

num_cases = 9
num_repetitions = 5

regexp = re.compile(r"""Test Count: (\d+)$
Train Count: (\d+)$
Dimensions: (\d+)$

GPU Distance Computation Time: (\d+\.\d+)$
GPU Sorting Time: (\d+\.\d+)$
GPU Time Taken: (\d+\.\d+)$

CPU Distance Computation Time: (\d+\.\d+)$
CPU Sorting Time: (\d+\.\d+)$
CPU Time Taken: (\d+\.\d+)""", re.M)

totalOutput = "" # latex string

headerRow = [
    ['', 'GPU','GPU','GPU', 'CPU', 'CPU', 'CPU'],
    ['Trial no.', 'Distance', 'Sorting', 'Total Time',
        'Distance', 'Sorting', 'Total Time']
]

f = open('latexTables.tex', 'w')

for case in range(1, num_cases+1):
    caseOutput = """\\subsection{Case %d}
\\begin{align*}
\\text{Test Count} &= %s \\\\
\\text{Train Count} &= %s \\\\
\\text{Dimensions} &= %s
\\end{align*}
"""
    matches = None # used for storing matches
    table = []
    row = []
    for count in range(num_repetitions):
        output = subprocess.check_output(
            ['CUDA-KNN', 'testdata/case%dtest.txt'%case, 'testdata/case%dtrain.txt'%case, '4'],
            universal_newlines = True
        )

        matches = regexp.match(output)
        if matches == None:
            f.write("ERROR:\n" + output + "\n")
            continue
            
        row = list(matches.groups())[3:] # remove the first few elements
        row.insert(0, count+1) # which trial
        table.append(row)

    caseOutput = caseOutput % ( case,
                                matches.groups()[0],
                                matches.groups()[1],
                                matches.groups()[2]
                              )
    
    table.append(['\\midrule Average', avg_col(table, 1), avg_col(table, 2), avg_col(table, 3), avg_col(table, 4), avg_col(table, 5), avg_col(table, 6)])
    speedup = float(avg_col(table, 6)) / float(avg_col(table, 3))
    
    caseOutput += "\n" + matrix2latex(table, None, headerRow=headerRow, format="%s").expandtabs(4)
    caseOutput += """
Speedup= $\\dfrac{\\text{Total CPU Time}}{\\text{Total GPU Time}}$ = $%.2f$x
""" % speedup
    f.write(caseOutput + "\n")

f.close()