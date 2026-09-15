using Assets.Scripts.IAJ.Unity.Util;
using UnityEngine;

namespace Assets.Scripts.IAJ.Unity.Movement.DynamicMovement
{
    public class DynamicWander : DynamicSeek
    {
        public float WanderOffset { get; set; }
        public float WanderRadius { get; set; }
        public float WanderRate { get; set; }
        public float WanderAngle { get; set; }

        public Vector3 CircleCenter { get; private set; }

        public GameObject DebugTarget { get; set; }

        public DynamicWander()
        {
            this.Target = new KinematicData();

            // ToDo Feel free to mess with these values and to initialize the rest of the variables, either here or in the manager
            this.WanderAngle = 0;
            this.WanderRadius = 5.0f;
            this.WanderRate = 0.2f;
            this.WanderOffset = 7.0f;
        }

        public override string Name
        {
            get { return "Wander"; }
        }


        public override MovementOutput GetMovement()
        {
            //This is how you call the helpers
            RandomHelper.RandomBinomial();
            MathHelper.ConvertOrientationToVector(Character.Orientation);

            //ToDo write here your code to implement the wander behavior

            if (this.DebugTarget != null)
            {
                this.DebugTarget.transform.position = this.Target.Position;
                
            }
            return base.GetMovement();
        }
    }
}
