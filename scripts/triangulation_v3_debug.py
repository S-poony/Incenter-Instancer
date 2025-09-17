import matplotlib
matplotlib.use("TkAgg") 

import math
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import triangulate
from shapely.geometry import Polygon, MultiPolygon
import os
import glob
import pandas as pd
import osmnx as ox

# ==================== INGEST ====================

place_name = "Hanoi, Vietnam"
tags = {"building": True} 

gdf = ox.features.features_from_place(place_name, tags=tags)
print(f"Loaded {len(gdf)} geometries from {place_name}")


# ==================== PROCESS (Shapely only) ====================
from shapely.geometry import Point
import numpy as np

MAXAREA = 20000  # base max area threshold
MINTRIANGLE = 30 # min area threshold for a triangle to be kept
MAX_TRIS = 100000  # safety cap

def split_triangle_centroid(tri):
    """Split triangle using centroid for better quality triangles."""
    coords = list(tri.exterior.coords)[:-1]
    if len(coords) != 3:
        return []
    centroid = tri.centroid
    cx, cy = centroid.x, centroid.y
    return [
        Polygon([coords[0], coords[1], (cx, cy)]),
        Polygon([coords[1], coords[2], (cx, cy)]),
        Polygon([coords[2], coords[0], (cx, cy)])
    ]

def generate_interior_points(poly, density=0.01):
    """Generate points inside the polygon for better triangulation."""
    if not isinstance(poly, Polygon) or poly.is_empty:
        return []
    
    bounds = poly.bounds
    min_x, min_y, max_x, max_y = bounds
    
    # Calculate spacing based on polygon size and desired density
    area = poly.area
    spacing = (area * density) ** 0.5
    
    interior_points = []
    x = min_x + spacing/2
    while x < max_x:
        y = min_y + spacing/2
        while y < max_y:
            point = Point(x, y)
            if poly.contains(point):
                interior_points.append((x, y))
            y += spacing
        x += spacing
    
    return interior_points

def simple_constrained_triangulate(poly):
    """Simpler triangulation that just filters basic triangulation results."""
    if not isinstance(poly, Polygon) or poly.is_empty:
        return []
    
    try:
        # Use basic Shapely triangulation
        basic_triangles = list(triangulate(poly))
        
        # Filter: keep triangles whose centroid is inside or touching the polygon
        valid_triangles = []
        for tri in basic_triangles:
            if tri.area > 0:
                centroid = tri.centroid
                # check centroid containment
                if poly.contains(centroid) or poly.boundary.distance(centroid) < 1e-10:
                    valid_triangles.append(tri)
        
        print(f"DEBUG: Basic triangulation: {len(basic_triangles)} → Filtered: {len(valid_triangles)}")
        return valid_triangles
        
    except Exception as e:
        print(f"Simple triangulation failed: {e}")
        return []

def calculate_shape_budget(geom, total_area, total_budget):
    """Calculate triangle budget for this shape based on its area proportion."""
    if total_area == 0:
        return 1
    area_proportion = geom.area / total_area
    shape_budget = max(1, int(area_proportion * total_budget))
    return shape_budget

def calculate_proportional_maxarea(geom, base_maxarea, total_area):
    """Calculate MAXAREA for this shape proportional to its size."""
    if total_area == 0:
        return base_maxarea
    
    area_ratio = geom.area / total_area
    scale_factor = max(0.1, area_ratio ** 0.5)
    proportional_maxarea = base_maxarea * scale_factor * 10
    return proportional_maxarea

def triangulate_geometry_simple(geom, shape_maxarea, shape_budget):
    """Simpler triangulation with better debugging."""
    triangles = []
    if geom is None or geom.is_empty:
        return triangles
    MINAREA= geom.area/MINTRIANGLE # TRIANGLE FAIT AU MOINS 2 PRCT DE LA TAILLE 
    
    print(f"DEBUG: Processing geometry with area {geom.area:.2f}")
    
    # Handle MultiPolygon vs Polygon
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    print(f"DEBUG: Processing {len(polys)} polygons")

    for poly_idx, poly in enumerate(polys):
        if not isinstance(poly, Polygon) or poly.is_empty:
            print(f"DEBUG: Skipping invalid polygon {poly_idx}")
            continue
        
        try:
            # Use simple constrained triangulation
            initial_triangles = simple_constrained_triangulate(poly)
            print(f"DEBUG: Got {len(initial_triangles)} initial triangles for polygon {poly_idx}")
            
            if not initial_triangles:
                print(f"DEBUG: No initial triangles for polygon {poly_idx}, trying basic triangulation")
                # Fallback to basic triangulation without filtering
                initial_triangles = list(triangulate(poly))
                print(f"DEBUG: Basic triangulation gave {len(initial_triangles)} triangles")
            
            stack = initial_triangles.copy()
            
        except Exception as e:
            print(f"Triangulation failed for polygon {poly_idx}: {e}")
            continue

        # Process stack until all triangles meet area constraints
        iterations = 0
        max_iterations = 1000
        
        while stack and len(triangles) < shape_budget and iterations < max_iterations:
            iterations += 1
            t = stack.pop()
            
            if t.area > shape_maxarea:
                print(f"DEBUG: Splitting triangle with area {t.area:.2f} (maxarea: {shape_maxarea:.2f})")
                new_triangles = split_triangle_centroid(t)
                for new_tri in new_triangles:
                    if new_tri.area > shape_maxarea:
                        stack.append(new_tri)
                    elif new_tri.area >= MINAREA:
                        triangles.append(new_tri)
                    # Remove the geometry containment check for now - it's too strict
            elif t.area >= MINAREA:
                triangles.append(t)
                print(f"DEBUG: Added triangle with area {t.area:.2f}")

            if len(triangles) >= shape_budget:
                print(f"DEBUG: Reached shape budget of {shape_budget}")
                break
        
        if iterations >= max_iterations:
            print(f"Warning: Hit max iterations for shape (area: {geom.area:.0f})")
        
        print(f"DEBUG: Polygon {poly_idx} contributed {len([t for t in triangles])} triangles")
    
    return triangles

# === CALCULATE PROPORTIONS ===
print("Calculating area proportions...")

total_area = sum(geom.area for geom in gdf.geometry if geom is not None and not geom.is_empty)
print(f"Total area across all shapes: {total_area:.0f}")

# === RUN WITH PROPORTIONAL ALLOCATION (SIMPLE VERSION) ===
triangles = []
shape_stats = []

print("DEBUG: Starting triangulation process...")

for i, geom in enumerate(gdf.geometry):
    if geom is None or geom.is_empty:
        print(f"DEBUG: Skipping empty geometry {i}")
        continue
    
    print(f"\nDEBUG: Processing shape {i}")
    
    shape_budget = calculate_shape_budget(geom, total_area, MAX_TRIS)
    shape_maxarea = calculate_proportional_maxarea(geom, MAXAREA, total_area)
    
    print(f"DEBUG: Shape {i} - Budget: {shape_budget}, MaxArea: {shape_maxarea:.2f}")
    
    shape_triangles = triangulate_geometry_simple(geom, shape_maxarea, shape_budget)
    triangles.extend(shape_triangles)
    
    area_proportion = geom.area / total_area * 100
    shape_stats.append({
        'shape_id': i,
        'area': geom.area,
        'area_proportion': area_proportion,
        'budget': shape_budget,
        'actual_triangles': len(shape_triangles),
        'maxarea': shape_maxarea
    })
    
    print(f"Shape {i}: Area={geom.area:.0f} ({area_proportion:.1f}%), "
          f"Budget={shape_budget}, Got={len(shape_triangles)} triangles")
    
    if len(triangles) >= MAX_TRIS:
        print(f"Reached global triangle limit at shape {i}")
        break

print(f"\nSIMPLE PROCESS COMPLETE: {len(triangles)} triangles across {len(shape_stats)} shapes")

# =============================== STORE DATA ===============================
def _triangle_vertices_from_polygon(poly):
    """Retourne les 3 sommets du triangle en tant que tuples (x,y)."""
    # On retire le point de fermeture (dernier == premier) puis on prend les 3 premiers
    coords = list(poly.exterior.coords)[:-1]
    # si pour une raison quelconque il y a >3 sommets (numérique), on récupère 3 uniques
    uniq = []
    for c in coords:
        if tuple(c) not in uniq:
            uniq.append(tuple(c))
        if len(uniq) == 3:
            break
    if len(uniq) != 3:
        return None
    return uniq  # [(x1,y1),(x2,y2),(x3,y3)]

def _side_length(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])

def compute_incenter_inradius_from_polygon(poly):
    """
    Calcule l'incenter et l'inradius d'un triangle shapely Polygon.
    Retourne dict ou None si triangle malformé.
    """
    verts = _triangle_vertices_from_polygon(poly)
    if verts is None:
        return None
    A, B, C = verts
    # conventions: a = |BC|, b = |CA|, c = |AB|
    a = _side_length(B, C)
    b = _side_length(C, A)
    c = _side_length(A, B)
    perim = a + b + c
    if perim == 0:
        return None
    # aire : on peut utiliser poly.area mais on recalcule pour robustesse
    x1,y1 = A; x2,y2 = B; x3,y3 = C
    area = abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)
    if area <= 1e-12:
        return None
    # incenter barycentrique pondéré par longueurs opposées
    ix = (a * x1 + b * x2 + c * x3) / perim
    iy = (a * y1 + b * y2 + c * y3) / perim
    s = perim / 2.0
    inradius = area / s
    return {
        "incenter_x": float(ix),
        "incenter_y": float(iy),
        "inradius": float(inradius),
        "area": float(area),
        "a": float(a), "b": float(b), "c": float(c)
    }

# Parcours des triangles existants (variable `triangles` déjà créée plus haut)
records = []
for tidx, tri in enumerate(triangles):
    try:
        res = compute_incenter_inradius_from_polygon(tri)
    except Exception as e:
        res = None
    if res is None:
        print ("Error: res is None, check compute_incenter_inradius_from_polygon")
    else:
        rec = {
            "triangle_id": tidx,
            "incenter_x": round(res["incenter_x"],2),
            "incenter_y": round(res["incenter_y"],2),
            "inradius": round(res["inradius"],2),
            "area": round(res["area"],2),
            "a": res["a"], "b": res["b"], "c": res["c"],
        }
        records.append(rec)

incenters_df = pd.DataFrame(records)

# Save to CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
# Fichier CSV à créer dans le même dossier
csv_path = os.path.join(script_dir, "incenters_inradii.csv")

incenters_df.to_csv(csv_path, index=False)
print(f"[INFO] Fichier écrit : {csv_path}")

# ==================== VISUALIZE WITH DEBUG INFO ====================
import matplotlib.pyplot as plt

print(f"DEBUG: Total triangles to visualize: {len(triangles)}")

if len(triangles) > 0:
    # Check triangle properties
    triangle_areas = [t.area for t in triangles]
    print(f"DEBUG: Triangle areas - Min: {min(triangle_areas):.2f}, Max: {max(triangle_areas):.2f}, Avg: {sum(triangle_areas)/len(triangle_areas):.2f}")
    
    # Check triangle bounds
    all_bounds = [t.bounds for t in triangles]
    if all_bounds:
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)
        print(f"DEBUG: Triangle bounds - X: {min_x:.2f} to {max_x:.2f}, Y: {min_y:.2f} to {max_y:.2f}")

# Also check original geometry bounds for comparison
if len(gdf) > 0:
    orig_bounds = gdf.total_bounds
    print(f"DEBUG: Original geometry bounds - X: {orig_bounds[0]:.2f} to {orig_bounds[2]:.2f}, Y: {orig_bounds[1]:.2f} to {orig_bounds[3]:.2f}")

# Render original polygons and triangulation result
fig, ax = plt.subplots(1, 2, figsize=(15, 8))

# Left: Original geometries
gdf.boundary.plot(ax=ax[0], color="black", linewidth=1)
ax[0].set_title(f"Original Shapes ({len(gdf)} geometries)")
ax[0].set_aspect('equal')

# Right: Triangulated result
triangle_count = 0
if len(triangles) > 0:
    for i, tri in enumerate(triangles):
        try:
            x, y = tri.exterior.xy
            ax[1].fill(x, y, edgecolor="black", facecolor="lightblue", alpha=0.6, linewidth=0.5)
            triangle_count += 1
        except Exception as e:
            print(f"DEBUG: Error plotting triangle {i}: {e}")
            continue
    
    ax[1].set_title(f"Triangulated Shapes ({triangle_count} triangles)")
    ax[1].set_aspect('equal')
    
    # Set same bounds as original for comparison
    if len(gdf) > 0:
        ax[1].set_xlim(ax[0].get_xlim())
        ax[1].set_ylim(ax[0].get_ylim())
else:
    ax[1].set_title("Triangulated Shapes (NO TRIANGLES GENERATED)")
    ax[1].text(0.5, 0.5, "No triangles to display", 
               horizontalalignment='center', verticalalignment='center', 
               transform=ax[1].transAxes, fontsize=12, color='red')

print(f"DEBUG: Successfully plotted {triangle_count} triangles")

plt.tight_layout()
plt.show()

# Additional debug: Show first few triangles' details
if len(triangles) > 0:
    print(f"\nDEBUG: First 3 triangles details:")
    for i, tri in enumerate(triangles[:3]):
        coords = list(tri.exterior.coords)
        print(f"Triangle {i}: Area={tri.area:.2f}, Coords={coords}")

# Check if triangulation is working at all
print(f"\nDEBUG: Shape processing summary:")
for i, geom in enumerate(gdf.geometry[:5]):  # Check first 5 shapes
    if geom is None or geom.is_empty:
        print(f"Shape {i}: Empty or None")
        continue
    print(f"Shape {i}: Area={geom.area:.2f}, Type={type(geom).__name__}")
    
    # Test basic triangulation on this shape
    try:
        from shapely.ops import triangulate
        basic_tris = list(triangulate(geom))
        print(f"  Basic triangulation: {len(basic_tris)} triangles")
    except Exception as e:
        print(f"  Basic triangulation failed: {e}")
        
    # Test our constrained triangulation
    try:
        constrained_tris = simple_constrained_triangulate(geom)
        print(f"  Constrained triangulation: {len(constrained_tris)} triangles")
    except Exception as e:
        print(f"  Constrained triangulation failed: {e}")