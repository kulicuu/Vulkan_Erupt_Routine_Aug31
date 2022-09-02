c = console.log.bind console
{ spawn, exec } = require 'child_process'

vert = exec ('glslc.exe ./shaders/terrain_100.vert -o ./spv/terrain_100.vert.spv'), (err) ->
    if err then c 'errors:', err

frag = exec ('glslc.exe ./shaders/terrain_100.frag -o ./spv/terrain_100.frag.spv'), (err) ->
    if err then c 'errors:', err

