# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 23:17:42 2022

@author: db_wi
"""

from PIL import Image,ImageDraw
def draw_square(o,x,y,val):
    c = [128 + 128*val, 128 - 128*val, 0]
    o[x][y] = c
    o[x+1][y] = c
    o[x][y+1] = c
    o[x+1][y+1] = c
    return o

def get_gray(val): return  (128 + 128*val, 128 + 128*val, 128 + 128*val)
def get_colour(val): return  (128 + 128*val, 128 - 128*val, 0)
    
def visualise(q_vars):
    # input layer
    output = Image.new('RGB', (2048+256,1280), (0,0,0))
    draw = ImageDraw.Draw(output)
    neuron_width = 3
    margin = 128
    
    #for inp in range(,len(q_vars),2):
    inp = -2
    print("Scanning layer %i" % inp)
    
    # kernel feed-forward layer
    for n in range(len(q_vars[inp].numpy())):
        if n % 10 == 0:
            print("Neuron %i" % n)
        links = q_vars[inp][n] > 0.5
        for l in range(len(q_vars[inp][n])):
            if not links[l]:
                continue
            val = q_vars[inp][n][l]
            c = get_gray(val)
            x1 = margin+n*neuron_width*2 + (neuron_width//2)
            y1 = margin+(inp-1)*neuron_width*16
            x2 = margin+l*neuron_width*2 + (neuron_width//2)
            y2 = margin+(inp+1)*neuron_width*16
            draw.line( (x1, y1, x2, y2), fill=c )    
            
    # bias layer
    for n in range(len(q_vars[inp+1].numpy())):
        val = q_vars[inp+1][n]
        c = get_colour(val)
        x1 = margin+n*neuron_width*2
        y1 = margin+(inp+1)*neuron_width*16
        draw.rectangle( (x1, y1, x1 + neuron_width, y1 + neuron_width), fill=c )    
    return output