using UnityEngine;
using UnityEditor;

// Custom inspector to show only relevant options depending on selected A* variant.
[CustomEditor(typeof(PathfindingManager))]
public class PathfindingManagerEditor : Editor
{
    SerializedProperty defaultPositions;
    SerializedProperty gridName;
    SerializedProperty neighbourhoodType;
    SerializedProperty aStarType;
    SerializedProperty openSetType;
    SerializedProperty closedSetType;
    SerializedProperty heuristics;
    SerializedProperty tieBreaking;
    SerializedProperty partialPath;
    SerializedProperty nodesPerFrame;

    // Gateway properties
    SerializedProperty preprocessOnStart;
    SerializedProperty preprocessKey;
    SerializedProperty visualizeGatewayClustersKey;
    SerializedProperty gatewayClusterMode;
    SerializedProperty gatewayRegionSize;

    void OnEnable()
    {
        defaultPositions = serializedObject.FindProperty("defaultPositions");
        gridName = serializedObject.FindProperty("gridName");
        neighbourhoodType = serializedObject.FindProperty("neighbourhoodType");
        aStarType = serializedObject.FindProperty("aStarType");
        openSetType = serializedObject.FindProperty("openSetType");
        closedSetType = serializedObject.FindProperty("closedSetType");
        heuristics = serializedObject.FindProperty("heuristics");
        tieBreaking = serializedObject.FindProperty("tieBreaking");
        partialPath = serializedObject.FindProperty("partialPath");
        nodesPerFrame = serializedObject.FindProperty("nodesPerFrame");

        preprocessOnStart = serializedObject.FindProperty("preprocessOnStart");
        preprocessKey = serializedObject.FindProperty("preprocessKey");
        visualizeGatewayClustersKey = serializedObject.FindProperty("visualizeGatewayClustersKey");
        gatewayClusterMode = serializedObject.FindProperty("gatewayClusterMode");
        gatewayRegionSize = serializedObject.FindProperty("gatewayRegionSize");
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        EditorGUILayout.LabelField("Grid Settings", EditorStyles.boldLabel);
        EditorGUILayout.PropertyField(gridName);
        EditorGUILayout.PropertyField(neighbourhoodType);
        EditorGUILayout.PropertyField(defaultPositions, true);

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Pathfinding Settings", EditorStyles.boldLabel);
        EditorGUILayout.PropertyField(aStarType);
        EditorGUILayout.PropertyField(heuristics);
        EditorGUILayout.PropertyField(tieBreaking);
        EditorGUILayout.PropertyField(partialPath);
        EditorGUILayout.PropertyField(nodesPerFrame);

        // Show open/closed set choices only for Vanilla A*
        var selected = (PathfindingManager.AStarType)aStarType.enumValueIndex;
        if (selected == PathfindingManager.AStarType.Vanilla)
        {
            EditorGUILayout.PropertyField(openSetType);
            EditorGUILayout.PropertyField(closedSetType);
        }
        else if (selected == PathfindingManager.AStarType.NodeArray)
        {
            EditorGUILayout.HelpBox("NodeArray uses a fixed Open/Closed set implementation (NodeRecordArray). Open/Closed options are not applicable.", MessageType.Info);
        }
        else if (selected == PathfindingManager.AStarType.GatewayAstar)
        {
            EditorGUILayout.HelpBox("Gateway A* uses its own preprocessing and gateway configuration.", MessageType.Info);
        }

        // Gateway-specific controls
        if (selected == PathfindingManager.AStarType.GatewayAstar)
        {
            EditorGUILayout.LabelField("Gateway A* Settings", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(preprocessOnStart);
            EditorGUILayout.PropertyField(preprocessKey);
            EditorGUILayout.PropertyField(visualizeGatewayClustersKey);
            EditorGUILayout.PropertyField(gatewayClusterMode);
            if ((PathfindingManager.GatewayClusterMode)gatewayClusterMode.enumValueIndex == PathfindingManager.GatewayClusterMode.FixedSize)
            {
                EditorGUILayout.PropertyField(gatewayRegionSize);
            }
        }

        // Draw remaining default inspector fields if any (optional)
        // This keeps the inspector minimal and relevant.

        serializedObject.ApplyModifiedProperties();
    }
}
