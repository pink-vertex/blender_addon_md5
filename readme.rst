====================================================================================
Blender Addon - Import and Export `id tech 4`_ ``*.md5mesh`` and ``*.md5anim`` files
====================================================================================

Installation:
=============

Download this repository as `zip-archive`_ and rename it to ``md5.zip``.
Go to Blender User Preferences -> Addons and click ``Install from File...`` and
select the zip-archive in the file browser which pops up.

Afterwards activate the addon under the category Import-Export or User.

Import:
=======

Choose from the Info Menu -> File Import -> ``*.md5mesh``.
In the file browser choose the file you want to import.

The ``*.md5mesh`` mesh does not store information about the
length of bones. You may edit the length of leaf bones **before**
you import an animation.

Then select the armature for which you want to import an animation.
Again choose from the Info Menu -> File Import -> ``*.md5anim`` and in
the file browser which pops up select the file you want to import.

Export:
=======

Put all the bones you want to import on one bone layer.
Using the default keymap you can do this by selecting a bone and press M. 

If you plan to export all bones and did not change any layer yet, 
you don't have to do anything.	

Select the armature of your model and choose Info Menu -> File Export -> ``*.md5mesh``.
The addon will gather and export all meshes which have an armature deform modifier
with the selected armature as target applied to them and are on an active scene layer.
In the file browser which pops up choose your bone layer. The default layer is 0.

To export an animation select the armature of your model and assign the action you want to export
to the armature. Afterwards choose Info Menu -> File Menu -> File Export -> ``*.md5anim``

If you plan to export an animation for an existing md5mesh, import the mesh and its armature
and use the latter to create your animation. The addon will store the bone indices in a custom
property named ``name_to_index`` of the armature.

Adding, or removing bones from the armature is **not** supported yet, unless you delete the 
custom property and export the new md5mesh and all of its animations.

Notes:
======

* For now, the addon has only been tested with blender version 2.77a and models of the game 
  Cube 2: Sauerbraten, which this addon was indented for.

* This addon originated from the `md5 addon`_ written by Nemyax.

* md5mesh files only support uv coordinates per vertex, while
  blender supports uv coordinates per face loop, which may result in 
  multiple uv coordinates per vertex.

  The addon will try to split the mesh along the borders of the uv islands so
  that each vertex can be assigned multiple uv coordinates if necessary.
  This will increase the vertex count of your model.

* The weights of your mesh should be normalized, which means for each vertice
  the sum of its weights should equal 1.0

* Although the ``*.md5anim`` format supports the export of single components of 
  the location vectors and rotation quaternions, the addon will always export all
  components using mask 63. 

.. _zip-archive: https://github.com/pink-vertex/blender_addon_md5/archive/Release.zip
.. _id tech 4: https://github.com/id-Software/DOOM-3 
.. _md5 addon: https://sourceforge.net/p/blenderbitsbobs/wiki/MD5%20exporter/
