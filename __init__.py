# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
	"name": "Id Tech 4 md5mesh and md5anim format",
	"author": "pink vertex",
	"version": (0, 1),
	"blender": (2, 77, 0),
	"location": "File -> Import-Export",
	"description": "Import-Export *.md5mesh and *.md5anim files",
	"warning": "",
	"wiki_url": "",
	"category": "Import-Export",
	}

if "bpy" in locals():
	import sys
	import importlib
	importlib.reload(sys.modules['md5.io_md5mesh'])
	importlib.reload(sys.modules['md5.io_md5anim'])

import bpy
from bpy.props import StringProperty, IntProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper

from .io_md5mesh import read_md5mesh, write_md5mesh
from .io_md5anim import read_md5anim, write_md5anim

class OT_IMPORT_MESH_md5mesh(bpy.types.Operator, ImportHelper):
	bl_idname = "import_mesh.md5"
	bl_label = "Import *.md5mesh Mesh"
	bl_options = {"REGISTER", "UNDO"}

	filename_ext = ".md5mesh"
	filter_glob = StringProperty(default="*.md5mesh", options={'HIDDEN'})

	def execute(self, context):
		read_md5mesh(self.filepath)
		return {"FINISHED"}


class OT_EXPORT_MESH_md5mesh(bpy.types.Operator, ExportHelper):
	bl_idname = "export_mesh.md5"
	bl_label = "Export *.md5mesh Mesh"
	bl_options = {"REGISTER", "UNDO"}

	filename_ext = ".md5mesh"
	filter_glob = StringProperty(default="*.md5mesh", options={'HIDDEN'})

	@classmethod
	def poll(cls, context):
		return (context.active_object and
				context.active_object.type == "ARMATURE")

	def execute(self, context):
		write_md5mesh(self.filepath, context.scene, context.active_object)
		return {"FINISHED"}

class OT_IMPORT_ANIM_md5anim(bpy.types.Operator, ImportHelper):
	bl_idname = "import_anim.md5"
	bl_label = "Import *.md5anim Animation"
	bl_options = {'REGISTER', "UNDO"}

	filename_ext = ".md5anim"
	filter_glob = StringProperty(default="*.md5anim", options={'HIDDEN'})

	@classmethod
	def poll(cls, context):
		return (context.active_object and
				context.active_object.type == "ARMATURE")

	def execute(self, context):
		bpy.ops.object.mode_set(mode="OBJECT")
		read_md5anim(self.filepath)
		return {'FINISHED'}

class OT_EXPORT_ANIM_md5anim(bpy.types.Operator, ExportHelper):
	bl_idname = "export_anim.md5"
	bl_label = "Export *.md5anim Anim"
	bl_options = {"REGISTER", "UNDO"}

	filename_ext = ".md5anim"
	filter_glob = StringProperty(default="*.md5anim", options={'HIDDEN'})
	bone_layer = IntProperty(name="BoneLayer", default=0)

	@classmethod
	def poll(cls, context):
		return (context.active_object and
			    context.active_object.type == "ARMATURE" and
				context.active_object.animation_data and
				context.active_object.animation_data.action)

	def execute(self, context):
		write_md5anim(self.filepath, context.scene, context.active_object, self.bone_layer)
		return {"FINISHED"}

reg_table = (
	[OT_IMPORT_MESH_md5mesh, bpy.types.INFO_MT_file_import, "MD5 Mesh (.md5mesh)"	  ],
	[OT_IMPORT_ANIM_md5anim, bpy.types.INFO_MT_file_import, "MD5 Animation (.md5anim)"],
	[OT_EXPORT_MESH_md5mesh, bpy.types.INFO_MT_file_export, "MD5 Mesh (.md5mesh)"     ],
	[OT_EXPORT_ANIM_md5anim, bpy.types.INFO_MT_file_export, "MD5 Animation (.md5anim)"]
)
	
def generate_menu_function(op_cls, description):
	def mnu_func(self, context):
		self.layout.operator(op_cls.bl_idname, text=description)
	return mnu_func

for row in reg_table:
	row[2] = generate_menu_function(row[0], row[2])

def register():
	for cls, mnu, mnu_func in reg_table:
		bpy.utils.register_class(cls)
		mnu.append(mnu_func)

def unregister():
	for cls, mnu, mnu_func in reg_table:
		mnu.remove(mnu_func)
		bpy.utils.unregister_class(cls)
